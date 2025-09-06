import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial, links
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
import scipy.stats

# -------------------------- 1. 环境配置与数据加载 --------------------------
# 设置中文字体与绘图样式
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 创建结果保存目录
results_dir = "scripts_v1.3_t1/result_question1"
os.makedirs(results_dir, exist_ok=True)

# 加载数据（需替换为实际数据路径，此处按题目隐含的"男胎检测数据"格式设计）
# 数据列名需与《CΜβ.pdf》附录1一致：A(样本序号)、B(孕妇代码)、C(年龄)、D(身高)、E(体重)、J(孕周)、K(BMI)、V(Y染色体浓度)等
df = pd.read_csv("scripts_v1.3_t1/dataA_Processed_v13.csv", encoding="utf-8")  # 若为Excel格式，替换为pd.read_excel("男胎检测数据.xlsx")

# 查看核心列是否存在（按《CΜβ.pdf》附录1校验）
required_cols = ["孕妇代码", "检测孕周", "孕妇BMI", "身高", "体重", "Y染色体浓度", "原始读段数", "在参考基因组上比对的比例", "重复读段的比例", "GC含量"]
assert all(col in df.columns for col in required_cols), "数据列名与《Cβ.pdf》附录1不匹配,请检查列名"


# -------------------------- 2. 数据预处理（按郝老师思路实现） --------------------------
def parse_gestational_age(s):
    """Parse gestational age from string format."""
    try:
        s = str(s).strip()
        if "w+" in s:
            w, d = s.replace("w", "").split("+")
            return int(w) + int(d) / 7
        elif "w" in s:
            return int(s.replace("w", ""))
        else:
            return np.nan
    except:
        return np.nan

def correct_y_concentration(p, eps=1e-4):
    """Correct Y concentration to avoid extremes."""
    p_numeric = pd.to_numeric(p, errors="coerce")
    return np.clip(p_numeric, eps, 1 - eps)

def verify_bmi(weight, height):
    """Verify BMI calculation."""
    weight_numeric = pd.to_numeric(weight, errors="coerce")
    height_numeric = pd.to_numeric(height, errors="coerce")
    calculated_bmi = weight_numeric / ((height_numeric / 100) ** 2)
    return calculated_bmi

# 2.1 孕周转换
df["孕周_连续值"] = df["检测孕周"].apply(parse_gestational_age)

# 2.2 Y染色体浓度修正
df["Y浓度_修正后"] = correct_y_concentration(df["Y染色体浓度"])

# 2.3 BMI一致性校验（标记异常值，不直接删除）
df["BMI_计算值"] = verify_bmi(df["体重"], df["身高"])
df["BMI_差值"] = np.abs(pd.to_numeric(df["孕妇BMI"], errors="coerce") - df["BMI_计算值"])
df["BMI_异常标记"] = (df["BMI_差值"] > 1).astype(int)  # 差值>1视为异常，后续模型控制

# 2.4 测序质量指标标准化（控制技术偏差）
quality_cols = ["在参考基因组上比对的比例", "重复读段的比例", "GC含量"]
for col in quality_cols:
    df[col + "_标准化"] = (pd.to_numeric(df[col], errors="coerce") - 
                          pd.to_numeric(df[col], errors="coerce").mean()) / \
                          pd.to_numeric(df[col], errors="coerce").std()

# 2.5 删除关键缺失值（保留有效样本）
df_clean = df.dropna(subset=["孕周_连续值", "Y浓度_修正后", "孕妇BMI", "孕妇代码"]).copy()
print(f"Original samples: {len(df)}, Cleaned samples: {len(df_clean)}")
print(f"Number of pregnant women: {df_clean['孕妇代码'].nunique()}")


# -------------------------- 3. 探索性数据分析（按郝老师思路可视化） --------------------------
# 3.1 单变量分布：Y浓度、孕周、BMI
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Y浓度分布
axes[0].hist(df_clean["Y浓度_修正后"], bins=30, color="#2E86AB", alpha=0.7, edgecolor="black")
axes[0].axvline(0.04, color="red", linestyle="--", linewidth=2, label="Threshold (4%)")
axes[0].set_title("Distribution of Corrected Y-chromosome Concentration", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Corrected Y-concentration")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# 孕周分布
axes[1].hist(df_clean["孕周_连续值"], bins=20, color="#A23B72", alpha=0.7, edgecolor="black")
axes[1].axvline(12, color="orange", linestyle="--", linewidth=2, label="Early/Mid-term Threshold (12w)")
axes[1].set_title("Distribution of Gestational Age", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Gestational Age (continuous)")
axes[1].set_ylabel("Frequency")
axes[1].legend()

# BMI分布
axes[2].hist(pd.to_numeric(df_clean["孕妇BMI"], errors="coerce"), bins=25, color="#F18F01", alpha=0.7, edgecolor="black")
axes[2].set_title("Distribution of Maternal BMI", fontsize=12, fontweight="bold")
axes[2].set_xlabel("BMI")
axes[2].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "单变量分布.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3.2 双变量关联：Y浓度与孕周、Y浓度与BMI
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Y浓度 vs 孕周（添加LOWESS平滑曲线）
sns.scatterplot(x="孕周_连续值", y="Y浓度_修正后", data=df_clean, ax=axes[0], 
                color="#2E86AB", alpha=0.6, s=30, edgecolor="none")
sns.regplot(x="孕周_连续值", y="Y浓度_修正后", data=df_clean, ax=axes[0], 
            scatter=False, color="red", lowess=True, line_kws={"linewidth":2})
corr_gest, p_gest = pearsonr(df_clean["孕周_连续值"], df_clean["Y浓度_修正后"])
axes[0].set_title(f"Y-concentration vs Gestational Age (Pearson r={corr_gest:.3f}, p<{p_gest:.4f})", 
                  fontsize=12, fontweight="bold")
axes[0].set_xlabel("Gestational Age (continuous)")
axes[0].set_ylabel("Corrected Y-concentration")
axes[0].axhline(0.04, color="orange", linestyle="--", linewidth=2, label="Threshold (4%)")
axes[0].legend()

# Y浓度 vs BMI（按BMI四分位分组）
df_clean["BMI_四分位"] = pd.qcut(pd.to_numeric(df_clean["孕妇BMI"], errors="coerce"), 
                               4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
sns.boxplot(x="BMI_四分位", y="Y浓度_修正后", data=df_clean, ax=axes[1], 
            palette=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"])
corr_bmi, p_bmi = pearsonr(pd.to_numeric(df_clean["孕妇BMI"], errors="coerce"), df_clean["Y浓度_修正后"])
axes[1].set_title(f"Y-concentration vs BMI Quartiles (Pearson r={corr_bmi:.3f}, p<{p_bmi:.4f})", 
                  fontsize=12, fontweight="bold")
axes[1].set_xlabel("BMI Quartile")
axes[1].set_ylabel("Corrected Y-concentration")
axes[1].axhline(0.04, color="orange", linestyle="--", linewidth=2, label="Threshold (4%)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "双变量关联分析.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3.3 交互效应：不同BMI分组下Y浓度-孕周趋势
fig, ax = plt.subplots(figsize=(12, 7))
for bmi_group, color in zip(["Q1 (Low)", "Q2", "Q3", "Q4 (High)"], ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]):
    group_data = df_clean[df_clean["BMI_四分位"] == bmi_group]
    # 拟合线性趋势
    z = np.polyfit(group_data["孕周_连续值"], group_data["Y浓度_修正后"], 1)
    p = np.poly1d(z)
    # 绘图
    sns.scatterplot(x="孕周_连续值", y="Y浓度_修正后", data=group_data, ax=ax, 
                    label=f"{bmi_group}(slope={z[0]:.4f})", color=color, alpha=0.6, s=30)
    ax.plot(group_data["孕周_连续值"], p(group_data["孕周_连续值"]), color=color, linewidth=2)

ax.set_title("Y-concentration vs Gestational Age Trend by BMI Group (Interaction)", fontsize=14, fontweight="bold")
ax.set_xlabel("Gestational Age (continuous)")
ax.set_ylabel("Corrected Y-concentration")
ax.axhline(0.04, color="orange", linestyle="--", linewidth=2, label="Threshold (4%)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "BMI-孕周交互效应.png"), dpi=300, bbox_inches="tight")
plt.close()


# -------------------------- 4. 模型构建（按郝老师Logit模型框架） --------------------------
# 4.1 定义变量（转换为数值型）
df_model = df_clean.copy()
df_model["BMI_数值"] = pd.to_numeric(df_model["孕妇BMI"], errors="coerce")
df_model["孕周_BMI交互"] = df_model["孕周_连续值"] * df_model["BMI_数值"]  # 交互项
df_model["比对比例_标准化"] = pd.to_numeric(df_model["在参考基因组上比对的比例_标准化"], errors="coerce")
df_model["重复读段比例_标准化"] = pd.to_numeric(df_model["重复读段的比例_标准化"], errors="coerce")
df_model["GC含量_标准化"] = pd.to_numeric(df_model["GC含量_标准化"], errors="coerce")

# 4.2 构建Logit模型（分数响应模型，按郝老师公式(5)(7)）
# 模型1：基础模型（无交互项）
model1 = glm(
    formula="Y浓度_修正后 ~ 孕周_连续值 + BMI_数值 + 比对比例_标准化 + 重复读段比例_标准化 + GC含量_标准化",
    data=df_model,
    family=Binomial(link=links.logit())  # Logit链接，符合郝老师思路
)
result1 = model1.fit(cov_type="cluster", cov_kwds={"groups": df_model["孕妇代码"]})  # 聚类稳健方差

# 模型2：扩展模型（含孕周×BMI交互项）
model2 = glm(
    formula="Y浓度_修正后 ~ 孕周_连续值 + BMI_数值 + 孕周_BMI交互 + 比对比例_标准化 + 重复读段比例_标准化 + GC含量_标准化",
    data=df_model,
    family=Binomial(link=links.logit())
)
result2 = model2.fit(cov_type="cluster", cov_kwds={"groups": df_model["孕妇代码"]})  # 聚类稳健方差

# 保存模型结果
with open(os.path.join(results_dir, "模型1_基础模型(无交互项).txt"), "w", encoding="utf-8") as f:
    f.write(result1.summary().as_text())
with open(os.path.join(results_dir, "模型2_扩展模型(含交互项).txt"), "w", encoding="utf-8") as f:
    f.write(result2.summary().as_text())

# 4.3 模型比较（似然比检验，按郝老师思路验证交互项必要性）
lr_stat = -2 * (result1.llf - result2.llf)  # 似然比统计量
lr_pvalue = 1 - scipy.stats.chi2.cdf(lr_stat, df=1)  # 自由度=1（交互项个数）
print("\n=== 模型比较（似然比检验）===")
print(f"似然比统计量：{lr_stat:.4f}")
print(f"p值:{lr_pvalue:.4f}")
print(f"结论：{'拒绝原假设(交互项显著，扩展模型更优）' if lr_pvalue < 0.05 else '接受原假设(交互项不显著，基础模型更优）'}")

# 保存模型比较结果
with open(os.path.join(results_dir, "模型比较_似然比检验.txt"), "w", encoding="utf-8") as f:
    f.write(f"似然比统计量：{lr_stat:.4f}\n")
    f.write(f"p值:{lr_pvalue:.4f}\n")
    f.write(f"结论：{'拒绝原假设(交互项显著，扩展模型更优）' if lr_pvalue < 0.05 else '接受原假设(交互项不显著，基础模型更优）'}\n")


# -------------------------- 5. 模型显著性检验与拟合优度（按郝老师要求） --------------------------
# 5.1 扩展模型（最优模型）系数显著性
print("\n=== 扩展模型（含交互项）系数显著性 ===")
coef_summary = result2.params.reset_index()
coef_summary.columns = ["变量", "系数"]
coef_summary["稳健标准误"] = result2.bse.values
coef_summary["Z值"] = result2.tvalues.values
coef_summary["p值"] = result2.pvalues.values
coef_summary["显著性"] = coef_summary["p值"].apply(lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else "ns")
print(coef_summary.to_string(index=False))
coef_summary.to_csv(os.path.join(results_dir, "扩展模型系数显著性.csv"), index=False, encoding="utf-8-sig")

# 5.2 拟合优度：McFadden伪R²（郝老师思路推荐指标）
def mcfadden_r2(logit_result):
    """计算McFadden伪R²"""
    ll_null = logit_result.null_deviance / (-2)
    ll_model = logit_result.llf
    print(f'Debug: ll_null = {ll_null}, ll_model = {ll_model}')
    r2 = 1 - (ll_model / ll_null)
    print(f'Debug: Calculated McFadden R² = {r2}')
    return r2

print("\n=== Goodness of Fit ===")
pseudo_r2 = mcfadden_r2(result2)
print(f"McFadden Pseudo R²: {pseudo_r2:.4f}")
print(f"Interpretation: {'Excellent (>0.4)' if pseudo_r2 > 0.4 else 'Good (0.2~0.4)' if pseudo_r2 > 0.2 else 'Fair (<0.2)' if pseudo_r2 > 0.2 else 'Fair (<0.2)'})")
print(f"r={corr_gest:.3f}, p<{p_gest:.4f}")
