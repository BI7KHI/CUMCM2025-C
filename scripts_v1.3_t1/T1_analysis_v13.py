import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial, links
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
import scipy.stats
from pygam import LogisticGAM, s, f
import io
from contextlib import redirect_stdout

# -------------------------- 1. 环境配置与数据加载 --------------------------
# 设置中文字体与绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 创建结果目录
results_dir = "results_v1.3"
os.makedirs(results_dir, exist_ok=True)

# 加载数据
df_clean = pd.read_csv("scripts_v1.3_t1/dataA_Processed_v13.csv", encoding="utf-8")
print(f"Original samples: {len(df_clean)}, Cleaned samples: {len(df_clean)}")

# -------------------------- 2. 数据预处理与探索性分析(EDA) --------------------------
# 确保Y浓度为数值类型
df_clean["Y浓度_修正后"] = pd.to_numeric(df_clean["Y染色体浓度"], errors="coerce")
df_clean.rename(columns={'G': '孕周_连续值'}, inplace=True)


# 打印孕妇数量
print(f"Number of pregnant women: {df_clean['孕妇代码'].nunique()}")

# -------------------------- 3. 特征工程与共线性诊断 --------------------------
# 标准化数值特征
scaler = StandardScaler()
features_to_scale = ["在参考基因组上比对的比例", "重复读段的比例", "GC含量"]
for feature in features_to_scale:
    df_clean[feature + "_标准化"] = scaler.fit_transform(df_clean[[feature]])

# 计算相关系数矩阵
correlation_matrix = df_clean[["孕周_连续值", "孕妇BMI"] + [f + "_标准化" for f in features_to_scale]].corr()
print("\n=== Correlation Matrix ===")
print(correlation_matrix)

# 计算VIF (方差膨胀因子)
vif_data = df_clean[["孕周_连续值", "孕妇BMI"] + [f + "_标准化" for f in features_to_scale]].dropna()
vif_data = sm.add_constant(vif_data)

vif_df = pd.DataFrame()
vif_df["Variable"] = vif_data.columns
vif_df["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]

print("\n=== Variance Inflation Factor (VIF) ===")
print(vif_df)

# -------------------------- 4. 模型构建与比较 --------------------------
# 定义模型中使用的变量
df_model = df_clean.copy()
# 基于中位数创建二元因变量
median_y = df_model["Y浓度_修正后"].median()
df_model["Y_binary"] = (df_model["Y浓度_修正后"] > median_y).astype(int)


# 诊断因变量
print("\n=== Dependent Variable Diagnostics ===")
print(df_model["Y_binary"].value_counts())

# 准备数据
X = df_model[["孕周_连续值", "孕妇BMI", "在参考基因组上比对的比例_标准化", "重复读段的比例_标准化", "GC含量_标准化"]]
X = sm.add_constant(X)
y = df_model["Y_binary"]

# --- 模型1: 基础模型 (无交互项) ---
logit_model1 = sm.Logit(y, X.drop(columns=["孕妇BMI"]))
result1 = logit_model1.fit_regularized(method='l1', alpha=0.1, disp=0)
with open(os.path.join(results_dir, "模型1_基础模型(无交互项).txt"), "w", encoding="utf-8") as f1:
    f1.write(result1.summary().as_text())

# --- 模型2: 扩展模型 (含交互项) ---
X["孕周_BMI交互"] = X["孕周_连续值"] * X["孕妇BMI"]
logit_model2 = sm.Logit(y, X)
result2 = logit_model2.fit_regularized(method='l1', alpha=0.1, disp=0)

# --- 模型比较: 似然比检验 ---
lr_stat = -2 * (result1.llf - result2.llf)
p_value_lrt = scipy.stats.chi2.sf(lr_stat, df=result2.df_model - result1.df_model)
print("\n=== 模型比较（似然比检验）===")
print(f"似然比统计量：{lr_stat:.4f}")
print(f"p值:{p_value_lrt:.4f}")
conclusion = "接受原假设(交互项不显著，基础模型更优）" if p_value_lrt > 0.05 else "拒绝原假设(交互项显著，扩展模型更优)"
print(f"结论：{conclusion}")

# --- 保存和打印扩展模型结果 ---
# 获取系数、p值等
summary_df = pd.DataFrame({
    "变量": result2.params.index,
    "系数": result2.params.values,
    "稳健标准误": result2.bse.values,
    "Z值": result2.tvalues.values,
    "p值": result2.pvalues.values
})
summary_df["显著性"] = summary_df["p值"].apply(lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns")))

print("\n=== 扩展模型（含交互项）系数显著性 ===")
print(summary_df.to_string(index=False))

# 将详细结果写入文件
with open(os.path.join(results_dir, "模型2_扩展模型(含交互项).txt"), "w", encoding="utf-8") as f2:
    f2.write("--- Coefficients and Significance ---\n")
    f2.write(summary_df.to_string(index=False))
    f2.write("\n\n--- Full Model Summary ---\n")
    f2.write(result2.summary().as_text())

# -------------------------- 5. 拟合优度与相关性分析 --------------------------
# print("\n=== Goodness of Fit ===")
# # McFadden's Pseudo R²
# mcfadden_r2 = result2.pseudo_rsquared(kind='mcf')
# print(f"McFadden Pseudo R²: {mcfadden_r2:.4f}")
# # 根据R²值进行解释
# if mcfadden_r2 > 0.4:
#     interpretation = "Excellent (>0.4)"
# elif mcfadden_r2 > 0.2:
#     interpretation = "Good (0.2~0.4)"
# else:
#     interpretation = "Fair (<0.2)"
# print(f"Interpretation: {interpretation}")

# # Durbin-Watson 检验
# dw_stat = durbin_watson(result2.resid)
# print(f"Durbin-Watson: {dw_stat:.2f}")

# # Pearson 相关性
# corr_gest, p_gest = pearsonr(df_model["孕周_连续值"], df_model["Y浓度_修正后"])
# # 注意：此处的r和p值是Y浓度与孕周的简单皮尔逊相关性，与整个模型的拟合优度是两个概念
# print(f"Pearson correlation (Y-conc vs Gest. Age): r={corr_gest:.3f}, p<{p_gest:.4f}")


# -------------------------- 6. GAM模型优化拟合 --------------------------
print("\n=== GAM Model Fitting (Optimization) ===")

# 准备数据，确保没有缺失值
gam_data = df_model.dropna(subset=[
    "Y_binary", "孕周_连续值", "孕妇BMI", 
    "在参考基因组上比对的比例_标准化", "重复读段的比例_标准化", "GC含量_标准化"
])

X_gam = gam_data[["孕周_连续值", "孕妇BMI", "在参考基因组上比对的比例_标准化", "重复读段的比例_标准化", "GC含量_标准化"]].values
y_gam = gam_data["Y_binary"].values

# 构建GAM模型：对孕周和BMI使用平滑项，其他为线性项
gam = LogisticGAM(
    s(0, n_splines=20, lam=0.6) +  # 孕周_连续值
    s(1, n_splines=20, lam=0.6) +  # 孕妇BMI
    f(2) +  # 在参考基因组上比对的比例_标准化
    f(3) +  # 重复读段的比例_标准化
    f(4)    # GC含量_标准化
).fit(X_gam, y_gam)

# 保存GAM模型摘要
with open(os.path.join(results_dir, "模型3_GAM模型.txt"), "w", encoding="utf-8") as f_out:
    f = io.StringIO()
    with redirect_stdout(f):
        gam.summary()
    summary_str = f.getvalue()
    f_out.write(summary_str)

# 打印并解释GAM结果
print("GAM Model Summary saved to '模型3_GAM模型.txt'")
gam_pseudo_r2 = gam.statistics_['pseudo_r2']['McFadden']
print(f"GAM McFadden Pseudo R²: {gam_pseudo_r2:.4f}")

# 可视化GAM的偏依赖图
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('GAM模型中关键变量的偏依赖图', fontsize=20)

titles = ["孕周", "孕妇BMI"]
for i, ax in enumerate(axes.flatten()):
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
    
    # 绘制偏依赖曲线和置信区间
    ax.plot(XX[:, i], pdep, color='blue', linewidth=3, label='偏依赖关系')
    ax.plot(XX[:, i], confi, c='red', ls='--', linewidth=2, label='95% 置信区间')
    
    # 添加地毯图 (Rug Plot)
    sns.rugplot(x=X_gam[:, i], ax=ax, color='black', height=0.05)
    
    ax.set_title(f'{titles[i]} 的偏依赖关系', fontsize=16)
    ax.set_xlabel(titles[i], fontsize=12)
    ax.set_ylabel("对数几率 (Log-odds)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(results_dir, "GAM_partial_dependence_optimized.png"), dpi=300)
print("\nGAM partial dependence plot saved to 'GAM_partial_dependence_optimized.png'")

plt.close()
print("GAM partial dependence plots saved.")


# -------------------------- 7. 可视化拟合曲线 --------------------------
print("\n=== Visualizing GAM Fit ===")

# 获取预测概率
pred_probs = gam.predict_proba(X_gam)

# 创建一个包含原始数据和预测概率的DataFrame，方便绘图
plot_df = pd.DataFrame({
    '孕周': X_gam[:, 0],
    '孕妇BMI': X_gam[:, 1],
    'Y_binary': y_gam,
    'Predicted_Prob': pred_probs
})

# 添加抖动以便观察
plot_df['Y_binary_jitter'] = plot_df['Y_binary'] + np.random.normal(0, 0.02, plot_df.shape[0])


# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('GAM 拟合曲线与实际数据点', fontsize=20)

# --- 孕周 vs 预测概率 ---
# 绘制实际数据点 (使用抖动以避免重叠)
sns.scatterplot(x='孕周', y='Y_binary_jitter', data=plot_df, ax=axes[0], alpha=0.3, color='gray', label='实际数据 (抖动)')
# 绘制拟合曲线
plot_df_sorted_gest = plot_df.sort_values('孕周')
axes[0].plot(plot_df_sorted_gest['孕周'], plot_df_sorted_gest['Predicted_Prob'], color='red', linewidth=2.5, label='GAM 拟合概率')
axes[0].set_title('孕周对Y染色体浓度的影响', fontsize=16)
axes[0].set_xlabel('孕周 (周)', fontsize=12)
axes[0].set_ylabel('Y染色体浓度 > 中位数的概率', fontsize=12)
axes[0].set_ylim(-0.1, 1.1)
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# --- 孕妇BMI vs 预测概率 ---
# 绘制实际数据点
sns.scatterplot(x='孕妇BMI', y='Y_binary_jitter', data=plot_df, ax=axes[1], alpha=0.3, color='gray', label='实际数据 (抖动)')
# 绘制拟合曲线
plot_df_sorted_bmi = plot_df.sort_values('孕妇BMI')
axes[1].plot(plot_df_sorted_bmi['孕妇BMI'], plot_df_sorted_bmi['Predicted_Prob'], color='red', linewidth=2.5, label='GAM 拟合概率')
axes[1].set_title('孕妇BMI对Y染色体浓度的影响', fontsize=16)
axes[1].set_xlabel('孕妇BMI', fontsize=12)
axes[1].set_ylabel('Y染色体浓度 > 中位数的概率', fontsize=12)
axes[1].set_ylim(-0.1, 1.1)
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(results_dir, "GAM_fitted_curve.png"), dpi=300)
print("GAM fitted curve plot saved to 'GAM_fitted_curve.png'")
plt.close()
