import pandas as pd
import numpy as np
from pygam import LogisticGAM, s, l, te
import os
import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')
import scipy.stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not installed, progress bar will not be shown. Run: pip install tqdm")
    def tqdm(iterable, **kwargs):
        return iterable

# ==============================================================================
# 0. 全局配置
# ==============================================================================
# 定义统一的路径
SOURCE_DATA_PATH = "Source_DATA/dataA.csv"
RESULTS_DIR = "result_question1"
PROCESSED_DATA_PATH = os.path.join(RESULTS_DIR, "dataA_Processed.csv")

# 确保结果目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# 1. 数据清洗与预处理
# ==============================================================================
def clean_data(source_path, dest_path):
    print("--- 步骤 1: 数据清洗与预处理 ---")
    def parse_gestational_age(s):
        try:
            s = str(s).strip()
            if "w+" in s:
                w, d = s.replace("w", "").split("+")
                return int(w) + int(d) / 7
            elif "w" in s:
                return int(s.replace("w", ""))
            return np.nan
        except:
            return np.nan

    df = pd.read_csv(source_path, encoding="utf-8")
    df["G"] = df["检测孕周"].apply(parse_gestational_age)
    df["Y浓度_修正后"] = pd.to_numeric(df["Y染色体浓度"], errors="coerce")
    
    # 核心列的缺失值必须被移除
    df_clean = df.dropna(subset=["G", "Y浓度_修正后", "孕妇BMI"])
    df_clean.to_csv(dest_path, index=False, encoding="utf-8")
    print(f"清洗后的数据已保存至: {dest_path}")
    return df_clean

# ==============================================================================
# 2. 环境配置 & 特征工程
# ==============================================================================
def feature_engineering(df):
    print("\n--- 步骤 2: 特征工程与共线性分析 ---")
    
    # --- 环境配置 ---
    def configure_chinese_font():
        try:
            fonts_dir = "fonts"
            if os.path.isdir(fonts_dir):
                for file_name in os.listdir(fonts_dir):
                    if file_name.lower().endswith((".ttf", ".otf")):
                        fm.fontManager.addfont(os.path.join(fonts_dir, file_name))
            candidate_families = ["Noto Sans CJK SC", "Source Han Sans SC", "SimHei"]
            installed_families = set(f.name for f in fm.fontManager.ttflist)
            for family in candidate_families:
                if family in installed_families:
                    plt.rcParams['font.family'] = [family]
                    plt.rcParams['font.sans-serif'] = [family]
                    return family
        except Exception:
            return None
        return None

    if configure_chinese_font():
        print("中文字体配置成功。")
    else:
        print("Warning: 未检测到可用中文字体。")
    plt.rcParams['axes.unicode_minus'] = False
    pd.set_option('display.max_columns', None)

    df.rename(columns={'G': '孕周_连续值'}, inplace=True)
    
    # --- 特征工程 ---
    scaler = StandardScaler()
    features_to_scale = ["在参考基因组上比对的比例", "重复读段的比例", "GC含量"]
    for feature in features_to_scale:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
        df[feature + "_标准化"] = scaler.fit_transform(df[[feature]].fillna(df[feature].median()))

    df["孕周_二次项"] = df["孕周_连续值"] ** 2
    df["BMI_二次项"] = df["孕妇BMI"] ** 2
    
    new_features = ["孕周_二次项", "BMI_二次项"]
    for feature in new_features:
        df[feature + "_标准化"] = scaler.fit_transform(df[[feature]])

    base_features = ["孕周_连续值", "孕妇BMI"] + [f + "_标准化" for f in features_to_scale]
    extended_features = base_features + [f + "_标准化" for f in new_features]
    
    # --- VIF分析 ---
    vif_data = df[extended_features].dropna()
    vif_data = sm.add_constant(vif_data)
    vif_df = pd.DataFrame()
    vif_df["Variable"] = vif_data.columns
    vif_df["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
    print("\n=== VIF 分析 (Top 5) ===")
    print(vif_df.sort_values('VIF', ascending=False).head())

    high_vif_features = vif_df[vif_df['VIF'] > 10]['Variable'].tolist()
    if 'const' in high_vif_features: high_vif_features.remove('const')
    
    filtered_features = [f for f in extended_features if f not in high_vif_features]
    print(f"\n移除高共线性特征后，剩余特征: {len(filtered_features)}个")
    
    return df, filtered_features

# ==============================================================================
# 3. GAM模型优化与评估
# ==============================================================================
def optimize_and_evaluate_gam(df, features):
    print("\n--- 步骤 3: GAM模型优化与评估 ---")
    
    # --- 准备数据 ---
    df["Y_binary"] = (df["Y浓度_修正后"] > df["Y浓度_修正后"].median()).astype(int)
    required_cols = ["Y_binary"] + features
    gam_data = df.dropna(subset=required_cols)
    X_gam = gam_data[features].values
    y_gam = gam_data["Y_binary"].values
    
    # --- 手动网格搜索 ---
    print("\n=== GAM 超参数搜索 ===")
    param_grid = {
        'lam': [0.01, 0.02, 0.05, 0.1, 0.3, 1.0],
        'n_splines': [20, 25, 30],
        'te_on': [False, True],
        'te_lam': [0.3, 1.0],
        'te_n_splines': [[8, 8], [10, 10]]
    }
    best_score = -1
    best_params = {}
    total_combinations = len(param_grid['lam']) * len(param_grid['n_splines']) * len(param_grid['te_on']) * len(param_grid['te_lam']) * len(param_grid['te_n_splines'])

    with tqdm(total=total_combinations, desc="Hyperparameter Search") as pbar:
        for lam in param_grid['lam']:
            for n_splines in param_grid['n_splines']:
                for te_on in param_grid['te_on']:
                    for te_lam in param_grid['te_lam']:
                        for te_ns in param_grid['te_n_splines']:
                            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                            fold_scores = []
                            for train_idx, val_idx in skf.split(X_gam, y_gam):
                                X_train, X_val = X_gam[train_idx], X_gam[val_idx]
                                y_train, y_val = y_gam[train_idx], y_gam[val_idx]
                                try:
                                    terms = s(0, n_splines=n_splines) + s(1, n_splines=n_splines)
                                    for i in range(2, X_gam.shape[1]):
                                        terms += l(i)
                                    if te_on and X_gam.shape[1] >= 2:
                                        terms += te(0, 1, n_splines=te_ns, lam=te_lam)
                                    gam_cv = LogisticGAM(terms, lam=lam).fit(X_train, y_train)
                                    if len(np.unique(y_val)) > 1:
                                        fold_scores.append(roc_auc_score(y_val, gam_cv.predict_proba(X_val)))
                                except Exception:
                                    fold_scores.append(np.nan)
                            mean_score = np.nanmean(fold_scores)
                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = {'lam': lam, 'n_splines': n_splines, 'te_on': te_on, 'te_lam': te_lam, 'te_n_splines': te_ns}
                            pbar.update(1)

    print(f"\n最佳AUC: {best_score:.4f} | 最佳参数: {best_params}")

    # --- 最终模型训练 ---
    final_terms = s(0, n_splines=best_params['n_splines']) + s(1, n_splines=best_params['n_splines'])
    for i in range(2, X_gam.shape[1]):
        final_terms += l(i)
    if best_params.get('te_on', False) and X_gam.shape[1] >= 2:
        final_terms += te(0, 1, n_splines=best_params.get('te_n_splines', [10, 10]), lam=best_params.get('te_lam', 0.6))

    best_gam = LogisticGAM(final_terms, lam=best_params['lam']).fit(X_gam, y_gam)
    
    # --- 保存结果 ---
    with open(os.path.join(RESULTS_DIR, "GAM_Optimal_Summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"--- Optimal GAM Model (AUC={best_score:.4f}) ---\n")
        f.write(f"Parameters: {best_params}\n\n")
        with redirect_stdout(io.StringIO()) as buf:
            best_gam.summary()
            f.write(buf.getvalue())
    print("最优GAM模型摘要已保存。")
    
    return best_gam, X_gam, y_gam

# ==============================================================================
# 4. 可视化
# ==============================================================================
def visualize_results(gam_model, X, y, features):
    print("\n--- 步骤 4: 结果可视化 ---")
    
    # --- 偏依赖图 ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    titles = features[:2]
    for i, ax in enumerate(axes.flatten()):
        XX = gam_model.generate_X_grid(term=i)
        pdep, confi = gam_model.partial_dependence(term=i, X=XX, width=0.95)
        ax.plot(XX[:, i], pdep, c='b', lw=2.5, label='偏依赖关系')
        ax.plot(XX[:, i], confi, c='r', ls='--', lw=1.8, label='95% 置信区间')
        sns.rugplot(x=X[:, i], ax=ax, color='k', height=0.05)
        ax.set_title(f'{titles[i]} 的偏依赖关系', fontsize=14)
        ax.set_xlabel(titles[i])
        ax.set_ylabel("对数几率 (Log-odds)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.suptitle('GAM模型关键变量的偏依赖图', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(RESULTS_DIR, "GAM_Partial_Dependence.png"), dpi=500, bbox_inches='tight')
    plt.close()
    print("偏依赖图已保存。")

    # --- 拟合曲线 ---
    plot_df = pd.DataFrame(X, columns=features)
    plot_df['Y_binary'] = y
    plot_df['Predicted_Prob'] = gam_model.predict_proba(X)
    plot_df['Y_binary_jitter'] = plot_df['Y_binary'] + np.random.normal(0, 0.02, len(plot_df))
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for i, (ax, feature_name) in enumerate(zip(axes, features[:2])):
        sns.scatterplot(x=feature_name, y='Y_binary_jitter', data=plot_df, ax=ax, alpha=0.25, s=12, label='实际数据 (抖动)')
        sorted_df = plot_df.sort_values(feature_name)
        ax.plot(sorted_df[feature_name], sorted_df['Predicted_Prob'], color='r', lw=2.2, label='GAM 拟合概率')
        ax.set_title(f'{feature_name}对Y染色体浓度的影响', fontsize=14)
        ax.set_ylabel('Y染色体浓度 > 中位数的概率')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.suptitle('GAM 拟合曲线与实际数据点', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(RESULTS_DIR, "GAM_Fitted_Curve.png"), dpi=500, bbox_inches='tight')
    plt.close()
    print("拟合曲线图已保存。")

# ==============================================================================
# 5. 置信度与拟合效果分析（通用）
# ==============================================================================
def analyze_performance_generic(probs, y, prefix="GAM"):
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    brier = brier_score_loss(y, probs)
    ll = log_loss(y, probs, labels=[0, 1])

    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1s = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
    best_idx = int(np.nanargmax(f1s)) if len(f1s) > 0 else 0
    best_thr = thresholds[best_idx - 1] if best_idx > 0 and best_idx - 1 < len(thresholds) else 0.5
    y_pred = (probs >= best_thr).astype(int)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Bootstrap CI
    n_boot = 200
    auc_samples, brier_samples = [], []
    for _ in tqdm(range(n_boot), desc=f"{prefix} Bootstrap CIs", leave=False):
        idx = np.random.randint(0, len(y), len(y))
        try:
            auc_samples.append(roc_auc_score(y[idx], probs[idx]))
            brier_samples.append(brier_score_loss(y[idx], probs[idx]))
        except Exception:
            continue
    def ci(a):
        arr = np.array(a)
        return np.nanpercentile(arr, [2.5, 97.5]) if arr.size > 0 else (np.nan, np.nan)
    auc_ci_low, auc_ci_high = ci(auc_samples)
    brier_ci_low, brier_ci_high = ci(brier_samples)

    # ROC
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y, probs)
    plt.title(f'{prefix} ROC (AUC={auc:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{prefix}_ROC.png'), dpi=500, bbox_inches='tight')
    plt.close()

    # PR
    plt.figure(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y, probs)
    plt.title(f'{prefix} PR (AP={ap:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{prefix}_PR.png'), dpi=500, bbox_inches='tight')
    plt.close()

    # Calibration
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy='quantile')
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='完美校准')
    plt.plot(mean_pred, frac_pos, 'o-', label=f'{prefix} 校准')
    plt.xlabel('预测概率'); plt.ylabel('实际阳性比例')
    plt.title(f'{prefix} 校准曲线')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{prefix}_Calibration.png'), dpi=500, bbox_inches='tight')
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['预测0', '预测1'], yticklabels=['实际0', '实际1'])
    plt.title(f'{prefix} 混淆矩阵@阈值={best_thr:.3f}\nAcc={acc:.3f}, F1={f1:.3f}')
    plt.xlabel('预测'); plt.ylabel('实际')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{prefix}_Confusion_Matrix.png'), dpi=500, bbox_inches='tight')
    plt.close()

    with open(os.path.join(RESULTS_DIR, f'{prefix}_Evaluation.txt'), 'w', encoding='utf-8') as f:
        f.write(f'=== {prefix} 模型评估 ===\n')
        f.write(f'AUC: {auc:.4f}  (95% CI: {auc_ci_low:.4f} ~ {auc_ci_high:.4f})\n')
        f.write(f'AP: {ap:.4f}\n')
        f.write(f'Brier: {brier:.4f}  (95% CI: {brier_ci_low:.4f} ~ {brier_ci_high:.4f})\n')
        f.write(f'LogLoss: {ll:.4f}\n')
        f.write(f'Best Threshold (F1-max): {best_thr:.4f}\n')
        f.write(f'Accuracy@thr: {acc:.4f}, F1@thr: {f1:.4f}\n')
        f.write('\nConfusion Matrix:\n')
        f.write(np.array2string(cm))

    return {
        'auc': auc, 'ap': ap, 'brier': brier,
        'best_thr': best_thr, 'acc': acc, 'f1': f1
    }

# 保留兼容的 GAM 专用封装
def analyze_performance(gam_model, X, y):
    print("\n--- 步骤 5: 置信度与拟合效果分析 ---")
    probs = gam_model.predict_proba(X)
    analyze_performance_generic(probs, y, prefix="GAM")
    print('评估报告与可视化已保存。')

# ==============================================================================
# 6. 其它算法对比与集成
# ==============================================================================
def train_and_compare_alternatives(X, y, features):
    print("\n--- 步骤 6: 其它算法对比与集成 ---")
    models = {
        'LR': Pipeline(steps=[('scaler', StandardScaler(with_mean=False)),  # 稀疏/稳健
                              ('clf', LogisticRegression(max_iter=2000, solver='lbfgs'))]),
        'RF': RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1),
        'GB': GradientBoostingClassifier(n_estimators=400, subsample=0.8, random_state=42)
    }

    summary_lines = []
    name_to_probs = {}

    for name, model in models.items():
        try:
            probs_oof = cross_val_predict(model, X, y, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
            auc = roc_auc_score(y, probs_oof)
            summary_lines.append(f'{name}: OOF AUC={auc:.4f}')
            name_to_probs[name] = probs_oof
        except Exception as e:
            summary_lines.append(f'{name}: 失败 ({e})')

    with open(os.path.join(RESULTS_DIR, 'Alternative_Models_Comparison.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    # 选择最佳替代模型并在全量数据上拟合
    best_name = max(name_to_probs, key=lambda k: roc_auc_score(y, name_to_probs[k])) if name_to_probs else None
    if best_name is None:
        print('替代模型均未成功。')
        return None, None

    best_model = models[best_name]
    best_model.fit(X, y)
    probs_full = best_model.predict_proba(X)[:, 1]
    analyze_performance_generic(probs_full, y, prefix=f'ALT_{best_name}')

    # 与GAM做简单集成（平均融合）
    print('尝试集成（GAM + 最佳替代模型）的概率平均...')
    return best_name, probs_full

# ==============================================================================
# 主执行流程
# ==============================================================================
if __name__ == "__main__":
    cleaned_df = clean_data(SOURCE_DATA_PATH, PROCESSED_DATA_PATH)
    featured_df, final_features = feature_engineering(cleaned_df)
    final_gam, X_final, y_final = optimize_and_evaluate_gam(featured_df, final_features)
    visualize_results(final_gam, X_final, y_final, final_features)
    analyze_performance(final_gam, X_final, y_final)

    # 训练替代模型、对比并可视化
    alt_name, alt_probs = train_and_compare_alternatives(X_final, y_final, final_features)
    if alt_probs is not None:
        # 融合
        gam_probs = final_gam.predict_proba(X_final)
        blend_probs = 0.5 * gam_probs + 0.5 * alt_probs
        analyze_performance_generic(blend_probs, y_final, prefix=f'Ensemble_GAM+{alt_name}')
    print("\n--- T1分析完成 ---")
