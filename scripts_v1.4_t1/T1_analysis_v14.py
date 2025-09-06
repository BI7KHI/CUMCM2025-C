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
from pygam import LogisticGAM, s, l, te
import io
from contextlib import redirect_stdout
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score
# from sklearn.model_selection import GridSearchCV # No longer needed

# --------- Chinese font configuration helper ---------
def configure_chinese_font():
    """
    Configure a Chinese-capable font for Matplotlib with robust fallbacks.
    - Tries common CJK families present on Linux (Noto/Source Han/SimHei/WenQuanYi).
    - Also loads any .ttf/.otf files under a project-level 'fonts' directory.
    Returns the chosen font family name or None if fallback to default.
    """
    try:
        # 1) Attempt to load any local fonts placed in project_root/fonts
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        fonts_dir = os.path.join(project_root, "fonts")
        if os.path.isdir(fonts_dir):
            for file_name in os.listdir(fonts_dir):
                if file_name.lower().endswith((".ttf", ".otf")):
                    try:
                        fm.fontManager.addfont(os.path.join(fonts_dir, file_name))
                    except Exception:
                        pass

        # 2) Try a prioritized list of common CJK font family names
        candidate_families = [
            "Noto Sans CJK SC",
            "Noto Sans SC",
            "Noto Sans S Chinese",
            "Source Han Sans SC",
            "Source Han Sans CN",
            "SimHei",
            "WenQuanYi Zen Hei",
            "Microsoft YaHei",
            "STHeiti",
            "PingFang SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]

        # Matplotlib sometimes returns a generic fallback path even if family not present.
        # Validate by checking family names present in font manager list.
        installed_families = set(f.name for f in fm.fontManager.ttflist)

        for family in candidate_families:
            try:
                if family in installed_families:
                    plt.rcParams['font.family'] = [family]
                    plt.rcParams['font.sans-serif'] = [family]
                    return family
            except Exception:
                continue
    except Exception:
        pass
    return None

# -------------------------- 1. 环境配置与数据加载 --------------------------
# 设置中文字体与绘图样式（带自动回退与本地fonts目录支持）
chosen_font = configure_chinese_font()
if chosen_font:
    print(f"Using Chinese font: {chosen_font}")
else:
    print("Warning: No CJK font found. Chinese text may not render correctly. "
          "Run 'python setup_chinese_fonts.py' at project root to install.")
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 创建结果目录
results_dir = "results_v1.4"
os.makedirs(results_dir, exist_ok=True)

# 加载数据
df_clean = pd.read_csv("scripts_v1.4_t1/dataA_Processed_v14.csv", encoding="utf-8")
print(f"Original samples: {len(df_clean)}, Cleaned samples: {len(df_clean)}")

# -------------------------- 2. 数据预处理与探索性分析(EDA) --------------------------
# 确保Y浓度为数值类型
df_clean["Y浓度_修正后"] = pd.to_numeric(df_clean["Y染色体浓度"], errors="coerce")
df_clean.rename(columns={'G': '孕周_连续值'}, inplace=True)


# 打印孕妇数量
print(f"Number of pregnant women: {df_clean['孕妇代码'].nunique()}")

# -------------------------- 3. 扩展特征工程与共线性诊断 --------------------------
# 标准化数值特征
scaler = StandardScaler()
features_to_scale = ["在参考基因组上比对的比例", "重复读段的比例", "GC含量"]
for feature in features_to_scale:
    df_clean[feature + "_标准化"] = scaler.fit_transform(df_clean[[feature]])

# 扩展特征工程：添加多项式特征和交互项
print("\n=== 扩展特征工程 ===")

# 1. 添加二次项（捕捉非线性关系）
df_clean["孕周_二次项"] = df_clean["孕周_连续值"] ** 2
df_clean["BMI_二次项"] = df_clean["孕妇BMI"] ** 2

# 2. 添加重要的交互项
df_clean["孕周_BMI交互"] = df_clean["孕周_连续值"] * df_clean["孕妇BMI"]
df_clean["孕周_GC交互"] = df_clean["孕周_连续值"] * df_clean["GC含量_标准化"]
df_clean["BMI_GC交互"] = df_clean["孕妇BMI"] * df_clean["GC含量_标准化"]

# 3. 添加比值特征（可能有生物学意义）
df_clean["比对质量比"] = df_clean["在参考基因组上比对的比例_标准化"] / (df_clean["重复读段的比例_标准化"] + 1e-8)

# 4. 标准化新特征
new_features = ["孕周_二次项", "BMI_二次项", "孕周_BMI交互", "孕周_GC交互", "BMI_GC交互", "比对质量比"]
for feature in new_features:
    df_clean[feature + "_标准化"] = scaler.fit_transform(df_clean[[feature]])

print(f"扩展后特征数量: {len(new_features) + len(features_to_scale) + 2} (原始: 孕周, BMI)")

# 计算相关系数矩阵（包含新特征）
extended_features = ["孕周_连续值", "孕妇BMI"] + [f + "_标准化" for f in features_to_scale + new_features]
correlation_matrix = df_clean[extended_features].corr()
print("\n=== 扩展特征相关系数矩阵（前5x5）===")
print(correlation_matrix.iloc[:5, :5])

# 计算VIF (方差膨胀因子) - 检查多重共线性
vif_data = df_clean[extended_features].dropna()
vif_data = sm.add_constant(vif_data)

vif_df = pd.DataFrame()
vif_df["Variable"] = vif_data.columns
vif_df["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]

print("\n=== 扩展特征 VIF 分析（前10个最高）===")
print(vif_df.sort_values('VIF', ascending=False).head(10))

# 过滤高共线性特征 (VIF > 10)
high_vif_features = vif_df[vif_df['VIF'] > 10]['Variable'].tolist()
if 'const' in high_vif_features:
    high_vif_features.remove('const')
    
if high_vif_features:
    print(f"\n高共线性特征 (VIF > 10): {high_vif_features}")
    # 保留原始特征，移除部分衍生特征
    filtered_features = [f for f in extended_features if f not in high_vif_features or f in ["孕周_连续值", "孕妇BMI"]]
    print(f"过滤后特征: {filtered_features}")
else:
    filtered_features = extended_features
    print("所有特征VIF均小于10，无需过滤")

# -------------------------- 4. 模型构建与比较 --------------------------
# 定义模型中使用的变量
df_model = df_clean.copy()
# 基于中位数创建二元因变量
median_y = df_model["Y浓度_修正后"].median()
df_model["Y_binary"] = (df_model["Y浓度_修正后"] > median_y).astype(int)


# 诊断因变量
print("\n=== Dependent Variable Diagnostics ===")
print(df_model["Y_binary"].value_counts())

# 准备数据（使用过滤后的扩展特征）
X = df_model[filtered_features]
X = sm.add_constant(X)
y = df_model["Y_binary"]
print(f"\n使用特征数量: {len(filtered_features)}")
print(f"特征列表: {filtered_features[:5]}..." if len(filtered_features) > 5 else f"特征列表: {filtered_features}")

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

# 准备GAM数据，使用扩展特征集
required_columns = ["Y_binary"] + filtered_features
gam_data = df_model.dropna(subset=required_columns)

X_gam = gam_data[filtered_features].values
y_gam = gam_data["Y_binary"].values

print(f"\nGAM数据形状: X_gam {X_gam.shape}, y_gam {y_gam.shape}")
print(f"GAM特征数量: {X_gam.shape[1]}")

# 动态构建GAM模型：前两个特征用平滑项，其余用线性项
print(f"构建动态GAM模型，特征数量: {X_gam.shape[1]}")
terms = []
# 前两个特征（通常是孕周和BMI）使用平滑项
terms.append(s(0, n_splines=20, lam=0.6))  # 第一个特征
if X_gam.shape[1] > 1:
    terms.append(s(1, n_splines=20, lam=0.6))  # 第二个特征

# 其余特征使用线性项
for i in range(2, X_gam.shape[1]):
    terms.append(l(i))

# 如果特征数量合适，添加关键交互项（孕周与BMI）
if X_gam.shape[1] >= 2:
    terms.append(te(0, 1, n_splines=[10, 10], lam=0.6))  # 孕周-BMI交互

gam = LogisticGAM(sum(terms[1:], terms[0])).fit(X_gam, y_gam)

# 保存GAM模型摘要
with open(os.path.join(results_dir, "模型3_GAM模型.txt"), "w", encoding="utf-8") as f_out:
    buf = io.StringIO()
    with redirect_stdout(buf):
        gam.summary()
    summary_str = buf.getvalue()
    f_out.write(summary_str)

# 打印并解释GAM结果
print("GAM Model Summary saved to '模型3_GAM模型.txt'")
gam_pseudo_r2 = gam.statistics_['pseudo_r2']['McFadden']
print(f"GAM McFadden Pseudo R²: {gam_pseudo_r2:.4f}")

# -------------------------- 6.1 GAM 增强超参数搜索 --------------------------
print("\n=== Enhanced GAM GridSearchCV for Hyperparameter Tuning ===")

# 定义更精细的超参数网格
param_grid = {
    'lam': np.logspace(-2, 2, 9),  # 更精细的lambda范围
    'n_splines': [12, 15, 20, 25, 30],  # 更多样条数选择
    'te_lam': np.logspace(-1, 1, 5) if X_gam.shape[1] >= 2 else [0.6],  # 交互项专用lambda
    'te_n_splines': [[8, 8], [10, 10], [12, 12]] if X_gam.shape[1] >= 2 else [[10, 10]]
}

best_score = -1
best_params = {}
cv_results = []

# 多目标优化：同时考虑AUC和拟合度
best_composite_score = -1
best_composite_params = {}

print(f"网格搜索规模: {len(param_grid['lam']) * len(param_grid['n_splines']) * len(param_grid['te_lam']) * len(param_grid['te_n_splines'])} 组合")

for lam in param_grid['lam']:
    for n_splines in param_grid['n_splines']:
        for te_lam in param_grid['te_lam']:
            for te_n_splines in param_grid['te_n_splines']:
                current_params = {
                    'lam': lam, 
                    'n_splines': n_splines,
                    'te_lam': te_lam,
                    'te_n_splines': te_n_splines
                }
                print(f"Testing params: lam={lam:.3f}, n_splines={n_splines}, te_lam={te_lam:.3f}")
                
                # 使用 K-Fold 交叉验证评估当前参数
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                fold_scores = []
                fold_pseudo_r2s = []
                
                for train_index, val_index in skf.split(X_gam, y_gam):
                    X_train, X_val = X_gam[train_index], X_gam[val_index]
                    y_train, y_val = y_gam[train_index], y_gam[val_index]

                    try:
                        # 动态构建模型
                        terms_cv = []
                        terms_cv.append(s(0, n_splines=n_splines, lam=lam))
                        if X_gam.shape[1] > 1:
                            terms_cv.append(s(1, n_splines=n_splines, lam=lam))
                        
                        for i in range(2, X_gam.shape[1]):
                            terms_cv.append(l(i, lam=lam))
                        
                        if X_gam.shape[1] >= 2:
                            terms_cv.append(te(0, 1, n_splines=te_n_splines, lam=te_lam))
                        
                        gam_cv = LogisticGAM(sum(terms_cv[1:], terms_cv[0])).fit(X_train, y_train)

                        if len(np.unique(y_val)) > 1:
                            auc_score = roc_auc_score(y_val, gam_cv.predict_proba(X_val))
                            pseudo_r2 = gam_cv.statistics_['pseudo_r2']['McFadden']
                            fold_scores.append(auc_score)
                            fold_pseudo_r2s.append(pseudo_r2)
                    except Exception as e:
                        fold_scores.append(np.nan)
                        fold_pseudo_r2s.append(np.nan)
                        continue
                
                mean_auc = np.nanmean(fold_scores)
                mean_pseudo_r2 = np.nanmean(fold_pseudo_r2s)
                
                # 复合评分：70% AUC + 30% Pseudo R²（平衡预测能力和拟合度）
                composite_score = 0.7 * mean_auc + 0.3 * mean_pseudo_r2 if not (np.isnan(mean_auc) or np.isnan(mean_pseudo_r2)) else -1
                
                cv_results.append({
                    'params': current_params, 
                    'mean_auc': mean_auc,
                    'mean_pseudo_r2': mean_pseudo_r2,
                    'composite_score': composite_score,
                    'std_auc': np.nanstd(fold_scores),
                    'std_pseudo_r2': np.nanstd(fold_pseudo_r2s)
                })

                if mean_auc > best_score:
                    best_score = mean_auc
                    best_params = current_params
                    
                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_composite_params = current_params

# 保存GridSearchCV结果
with open(os.path.join(results_dir, "GAM_GridSearch_results.txt"), "w", encoding="utf-8") as f_grid:
    f_grid.write("--- Enhanced GAM GridSearchCV Results ---\n")
    f_grid.write(f"Best AUC score: {best_score:.4f}\n")
    f_grid.write(f"Best AUC parameters: {best_params}\n\n")
    f_grid.write(f"Best composite score: {best_composite_score:.4f}\n")
    f_grid.write(f"Best composite parameters: {best_composite_params}\n\n")
    f_grid.write("--- CV Results (Top 10 by composite score) ---\n")
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_sorted = cv_results_df.sort_values('composite_score', ascending=False).head(10)
    f_grid.write(cv_results_sorted.to_string())
print("Enhanced GAM GridSearchCV results saved to 'GAM_GridSearch_results.txt'")

# 使用复合最佳参数训练最终模型（平衡预测能力和拟合度）
print(f"\nTraining final model with best composite params: {best_composite_params}")
final_terms = []
final_terms.append(s(0, n_splines=best_composite_params.get('n_splines', 20), lam=best_composite_params.get('lam', 0.6)))
if X_gam.shape[1] > 1:
    final_terms.append(s(1, n_splines=best_composite_params.get('n_splines', 20), lam=best_composite_params.get('lam', 0.6)))

for i in range(2, X_gam.shape[1]):
    final_terms.append(l(i, lam=best_composite_params.get('lam', 0.6)))

if X_gam.shape[1] >= 2:
    final_terms.append(te(0, 1, n_splines=best_composite_params.get('te_n_splines', [10, 10]), lam=best_composite_params.get('te_lam', 0.6)))

best_gam = LogisticGAM(sum(final_terms[1:], final_terms[0])).fit(X_gam, y_gam)

print(f"Final model Pseudo R²: {best_gam.statistics_['pseudo_r2']['McFadden']:.4f}")
print(f"Final model AIC: {best_gam.statistics_['AIC']:.4f}")


# 保存最佳GAM模型摘要
with open(os.path.join(results_dir, "模型3_GAM模型_Optimized.txt"), "w", encoding="utf-8") as f_out:
    buf = io.StringIO()
    with redirect_stdout(buf):
        best_gam.summary()
    summary_str = buf.getvalue()
    f_out.write(summary_str)
print("Optimized GAM Model Summary saved to '模型3_GAM模型_Optimized.txt'")


# -------------------------- 6.2 GAM K折交叉验证 --------------------------
def evaluate_gam_with_cv(X: np.ndarray, y: np.ndarray, gam_model, n_splits: int = 5, random_state: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    auc_list, acc_list, ll_list, f1_list = [], [], [], []
    fold_idx = 0
    per_fold_lines = []

    for train_index, val_index in skf.split(X, y):
        fold_idx += 1
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 每折拟合传入的最佳模型结构
        fold_gam = gam_model.fit(X_train, y_train)

        prob_val = fold_gam.predict_proba(X_val)
        prob_val = np.clip(prob_val, 1e-15, 1 - 1e-15)
        pred_val = (prob_val >= 0.5).astype(int)

        # 计算指标
        try:
            if len(np.unique(y_val)) > 1:
                auc = roc_auc_score(y_val, prob_val)
            else:
                auc = np.nan
        except Exception:
            auc = np.nan

        acc = accuracy_score(y_val, pred_val)
        try:
            ll = log_loss(y_val, prob_val)
        except Exception:
            ll = np.nan
        f1 = f1_score(y_val, pred_val)

        auc_list.append(auc)
        acc_list.append(acc)
        ll_list.append(ll)
        f1_list.append(f1)

        per_fold_lines.append(
            f"Fold {fold_idx}: AUC={auc if not np.isnan(auc) else 'nan':>6}, "
            f"Acc={acc:.4f}, LogLoss={ll if not np.isnan(ll) else 'nan':>6}, F1={f1:.4f}"
        )

    def nanmean_std(values: list):
        arr = np.array(values, dtype=float)
        return np.nanmean(arr), np.nanstd(arr)

    auc_mean, auc_std = nanmean_std(auc_list)
    acc_mean, acc_std = nanmean_std(acc_list)
    ll_mean, ll_std = nanmean_std(ll_list)
    f1_mean, f1_std = nanmean_std(f1_list)

    lines = []
    lines.append("=== GAM K折交叉验证 (StratifiedKFold) ===")
    lines.append(f"样本量: {len(y)}, 正类比例: {y.mean():.3f}, 折数: {n_splits}")
    lines.append("")
    lines.extend(per_fold_lines)
    lines.append("")
    lines.append(
        f"AUC: mean={auc_mean:.4f} std={auc_std:.4f} (nan表示该折仅单一类别)"
    )
    lines.append(
        f"Accuracy: mean={acc_mean:.4f} std={acc_std:.4f}"
    )
    lines.append(
        f"LogLoss: mean={ll_mean:.4f} std={ll_std:.4f}"
    )
    lines.append(
        f"F1: mean={f1_mean:.4f} std={f1_std:.4f}"
    )

    return "\n".join(lines)


cv_report = evaluate_gam_with_cv(X_gam, y_gam, best_gam, n_splits=5, random_state=42)
with open(os.path.join(results_dir, "GAM_CV_results_Optimized.txt"), "w", encoding="utf-8") as fcv:
    fcv.write(cv_report)
print("Optimized GAM 5-fold CV results saved to 'GAM_CV_results_Optimized.txt'")


# 可视化GAM的偏依赖图
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('优化后GAM模型关键变量的偏依赖图', fontsize=20)

titles = ["孕周", "孕妇BMI"]
# 主效应
for i, ax in enumerate(axes.flatten()):
    XX = best_gam.generate_X_grid(term=i)
    pdep, confi = best_gam.partial_dependence(term=i, X=XX, width=0.95)

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


# -------------------------- 7. 可视化拟合曲线 --------------------------
print("\n=== Visualizing GAM Fit ===")

# 获取预测概率
pred_probs = best_gam.predict_proba(X_gam)

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
