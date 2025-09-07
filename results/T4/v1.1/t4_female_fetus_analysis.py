import os
import re
import json
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             classification_report, confusion_matrix, recall_score, precision_score, f1_score)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.mixture import GaussianMixture
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
import argparse
from typing import Optional

# ---------------------- 配置 ----------------------
BASE_DIR = r"C:\Users\Admin\Desktop\CUMCM2025-C\CUMCM2025-C"
DATA_A = os.path.join(BASE_DIR, 'Source_DATA', 'dataA.csv')
DATA_B = os.path.join(BASE_DIR, 'Source_DATA', 'dataB.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'T4')
os.makedirs(RESULTS_DIR, exist_ok=True)
V1_DIR = os.path.join(BASE_DIR, 'scripts', 'T4', 'v1.0')
os.makedirs(V1_DIR, exist_ok=True)

# ---------------------- 工具函数 ----------------------

def parse_gestational_age(s: str):
    """解析形如 '13w+6'、'13w' 的检测孕周，返回 (周_浮点, 天数)"""
    if pd.isna(s):
        return np.nan, np.nan
    s = str(s).strip()
    m = re.match(r"(\d+)w(?:\+(\d+))?", s)
    if m:
        w = int(m.group(1))
        d = int(m.group(2)) if m.group(2) else 0
        return w + d/7.0, w*7 + d
    # 兜底：如果是纯数字
    try:
        v = float(s)
        return v, int(round(v*7))
    except Exception:
        return np.nan, np.nan


def read_and_unify(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    # 去除列名空格
    df.columns = [str(c).strip() for c in df.columns]
    # 统一关键列名映射
    rename_map = {
        '唯一比对的读段数': '唯一比对的读段数',
        '唯一比对的读段数  ': '唯一比对的读段数',
        '13号染色体的Z值': 'Z13',
        '18号染色体的Z值': 'Z18',
        '21号染色体的Z值': 'Z21',
        'X染色体的Z值': 'ZX',
        'Y染色体的Z值': 'ZY',
        'Y染色体浓度': 'Y浓度',
        'X染色体浓度': 'X浓度',
        'GC含量': 'GC_total',
        '13号染色体的GC含量': 'GC13',
        '18号染色体的GC含量': 'GC18',
        '21号染色体的GC含量': 'GC21',
        '在参考基因组上比对的比例': '比对比例',
        '重复读段的比例': '重复比例',
        '被过滤掉读段数的比例': '过滤比例',
        '孕妇BMI': 'BMI',
        '检测孕周': '孕周文本',
        '孕妇代码': '孕妇代码',
        '原始读段数': '原始读段数',
        '唯一比对的读段数': '唯一比对的读段数',
        '染色体的非整倍体': 'AB',
        '胎儿是否健康': '胎儿是否健康'
    }
    df = df.rename(columns=rename_map)

    # 解析孕周
    if '孕周文本' in df.columns:
        parsed = df['孕周文本'].apply(parse_gestational_age)
        df['孕周_float'] = parsed.map(lambda x: x[0])
        df['孕周_天数'] = parsed.map(lambda x: x[1])

    # 补齐缺失列
    for col in ['Z13','Z18','Z21','ZX','ZY','Y浓度','X浓度','GC_total','GC13','GC18','GC21',
                '比对比例','重复比例','过滤比例','BMI','原始读段数','唯一比对的读段数','孕妇代码','AB','胎儿是否健康','孕周_float','孕周_天数']:
        if col not in df.columns:
            df[col] = np.nan

    return df


def filter_female(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    使用 Y浓度 的双峰聚类自动阈值，选择女胎样本（低 Y浓度簇）；
    若Y浓度缺失，默认视为女胎（因为题设为女胎判定场景）。
    返回：过滤后的df 和 统计信息。
    """
    stats = {}
    yvals = df['Y浓度'].dropna().values.reshape(-1, 1)
    if len(yvals) >= 20:
        gmm = GaussianMixture(n_components=2, random_state=42)
        y_df_fit = df[['Y浓度']].dropna()
        gmm.fit(y_df_fit)
        means = gmm.means_.ravel()
        low_idx = int(np.argmin(means))
        probs = gmm.predict_proba(df[['Y浓度']].fillna(-1e6))
        female_mask = (np.argmax(probs, axis=1) == low_idx)
        thr_est = float(means[low_idx] + 3 * np.sqrt(gmm.covariances_.ravel()[low_idx]))
        stats.update({'method': 'GMM', 'means': means.tolist(), 'thr_est': thr_est})
    else:
        # 数据不足：采用经验阈值（分位数）
        non_missing = df['Y浓度'].dropna()
        if len(non_missing) > 0:
            thr_est = float(np.nanpercentile(non_missing, 40))
            female_mask = (df['Y浓度'].isna()) | (df['Y浓度'] <= thr_est)
            stats.update({'method': 'quantile40', 'thr_est': thr_est})
        else:
            # 完全缺失：全部视为女胎
            female_mask = np.ones(len(df), dtype=bool)
            stats.update({'method': 'all_female_due_to_missing'})
    filtered = df[female_mask].copy()
    stats.update({'total': int(len(df)), 'female_selected': int(len(filtered))})
    return filtered, stats


def build_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    # 标签：AB列包含 T13/T18/T21 任一即异常
    ab = df['AB'].fillna('')
    y = ab.str.contains('T13|T18|T21', regex=True).astype(int)

    # 工程化特征
    df['唯一比对率'] = df['唯一比对的读段数'] / df['原始读段数']
    feat_cols = [
        'Z13','Z18','Z21','ZX',
        'GC_total','GC13','GC18','GC21',
        '原始读段数','比对比例','重复比例','唯一比对率','过滤比例',
        'BMI','孕周_float'
    ]
    X = df[feat_cols].copy()
    # 数值化
    for c in feat_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(X.median(numeric_only=True))
    return X, y


# ---------------------- 主流程 ----------------------



def main(target_recall: float = 0.90, test_size: float = 0.2, seed: int = 42,
         results_dir: Optional[str] = None,
         calibrate: bool = False,
         calib_size: float = 0.2,
         calib_method: str = 'sigmoid'):
    np.random.seed(seed)
    if results_dir is not None:
        global RESULTS_DIR
        RESULTS_DIR = results_dir
        os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) 读取与合并
    dfa = read_and_unify(DATA_A)
    dfb = read_and_unify(DATA_B)
    raw = pd.concat([dfa, dfb], ignore_index=True)

    # 2) 仅保留女胎样本并构建特征与标签
    female_df, female_stats = filter_female(raw)
    X, y = build_features(female_df)
    groups = female_df['孕妇代码'].astype(str).fillna('NA')

    # 3) 按孕妇代码分组划分训练/测试
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 4) 模型与超参（GroupKFold 交叉验证）
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=200, class_weight='balanced', solver='liblinear', random_state=seed))
    ])
    gkf = GroupKFold(n_splits=5)
    param_grid = {
        'clf__C': [0.1, 1.0, 3.0, 10.0],
        'clf__penalty': ['l1', 'l2']
    }
    gs = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', cv=gkf, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train, groups=groups.iloc[train_idx])
    best_model = gs.best_estimator_
    # 可选概率校准（Platt / Isotonic），在训练集内部保留校准子集，并保持孕妇分组不泄漏
    if calibrate and 0.0 < calib_size < 0.9:
        gss_cal = GroupShuffleSplit(n_splits=1, test_size=calib_size, random_state=seed)
        tr_sub_idx, calib_idx = next(gss_cal.split(X_train, y_train, groups.iloc[train_idx]))
        X_tr_sub, y_tr_sub = X_train.iloc[tr_sub_idx], y_train.iloc[tr_sub_idx]
        X_calib, y_calib = X_train.iloc[calib_idx], y_train.iloc[calib_idx]
        base = clone(best_model)
        base.fit(X_tr_sub, y_tr_sub)
        best_model = CalibratedClassifierCV(base_estimator=base, cv='prefit', method=calib_method)
        best_model.fit(X_calib, y_calib)

    # 5) 训练集阈值选择（PR）
    prob_train = best_model.predict_proba(X_train)[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, prob_train)
    prec_aligned = prec[:-1]
    rec_aligned = rec[:-1]
    f1_aligned = 2 * prec_aligned * rec_aligned / (prec_aligned + rec_aligned + 1e-12)
    if len(f1_aligned) > 0:
        best_idx = int(np.nanargmax(f1_aligned))
        best_thr = float(thr[best_idx]) if len(thr) > 0 else 0.5
    else:
        best_thr = 0.5
    cand_idx = np.where(rec_aligned >= target_recall)[0]
    if len(cand_idx) > 0 and len(thr) > 0:
        best_in_cand = np.argmax(f1_aligned[cand_idx])
        idx_r = int(cand_idx[best_in_cand])
        rec90_thr = float(thr[idx_r])
    elif len(thr) > 0 and len(rec_aligned) > 0:
        idx_r = int(np.argmin(np.abs(rec_aligned - target_recall)))
        rec90_thr = float(thr[idx_r])
    else:
        rec90_thr = best_thr

    # 6) 测试集评估
    prob_test = best_model.predict_proba(X_test)[:, 1]
    y_pred = (prob_test >= best_thr).astype(int)
    y_pred_rec90 = (prob_test >= rec90_thr).astype(int)

    metrics = {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, prob_test)) if len(np.unique(y_test)) > 1 else None,
        'best_threshold': float(best_thr),
        'classification_report': classification_report(y_test, y_pred, zero_division=0)
    }
    metrics_rec90 = {
        'precision': float(precision_score(y_test, y_pred_rec90, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_rec90, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_rec90, zero_division=0)),
        'roc_auc': metrics['roc_auc'],
        'threshold': float(rec90_thr),
        'classification_report': classification_report(y_test, y_pred_rec90, zero_division=0)
    }

    # 保存筛选后的样本，便于复核
    female_df.to_csv(os.path.join(RESULTS_DIR, 'female_samples.csv'), index=False, encoding='utf-8-sig')

    # 可选：导出一棵浅树的可解释规则
    tree = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=seed)
    tree.fit(X_train, y_train)
    rules = export_text(tree, feature_names=list(X.columns))
    with open(os.path.join(RESULTS_DIR, 'tree_rules.txt'), 'w', encoding='utf-8') as rf:
        rf.write(rules)

    # 下方绘图/保存依赖的变量：rec, prec, thr, best_thr, rec90_thr, metrics, metrics_rec90, prob_test, y_pred, y_pred_rec90, test_idx, female_df
    # 因此无需改动原有绘图与保存代码，仅修正缩进不一致的行
    # PR（训练集）并标注两种阈值
    plt.figure(figsize=(5,4), dpi=150)
    plt.plot(rec, prec, label='PR (train)')
    # 找到与best_thr、rec90_thr相近的点用于标注
    if len(thr)>0:
        # 对应阈值索引：thr与prec/rec的长度差1，使用前面对齐的索引
        try:
            idx_best = np.argmin(np.abs(thr - best_thr))
            plt.scatter(rec_aligned[idx_best], prec_aligned[idx_best], c='g', label=f'F1最优阈值≈{best_thr:.3f}')
        except Exception:
            pass
        try:
            idx_r = np.argmin(np.abs(thr - rec90_thr))
            plt.scatter(rec_aligned[idx_r], prec_aligned[idx_r], c='r', label=f'召回优先阈值≈{rec90_thr:.3f}')
        except Exception:
            pass
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (Train)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pr_curve_train.svg'))
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, prob_test)
    plt.figure(figsize=(5,4), dpi=150)
    plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}" if metrics['roc_auc'] is not None else 'AUC=N/A')
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    # 混淆矩阵（F1最优）
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4,4), dpi=150)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['正常','异常'])
    ax.set_yticklabels(['正常','异常'])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    ax.set_title('Confusion Matrix (Female)')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.svg'))
    plt.close(fig)
    # 混淆矩阵（召回优先）
    cm2 = confusion_matrix(y_test, y_pred_rec90)
    fig2, ax2 = plt.subplots(figsize=(4,4), dpi=150)
    ax2.imshow(cm2, cmap='Oranges')
    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    ax2.set_xticklabels(['正常','异常'])
    ax2.set_yticklabels(['正常','异常'])
    for (i,j), v in np.ndenumerate(cm2):
        ax2.text(j, i, str(v), ha='center', va='center')
    ax2.set_title('Confusion Matrix (Recall-priority)')
    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_recall.svg'))
    plt.close(fig2)

    # 保存指标JSON与测试集预测输出
    # 保存模型与阈值配置
    try:
        import joblib
        joblib.dump(best_model, os.path.join(RESULTS_DIR, 'model.joblib'))
        with open(os.path.join(RESULTS_DIR, 'thresholds.json'), 'w', encoding='utf-8') as tf:
            json.dump({'best_threshold_f1': float(best_thr), 'recall_priority_threshold': float(rec90_thr), 'target_recall': float(target_recall)}, tf, ensure_ascii=False, indent=2)
    except Exception as e:
        print('Warning: failed to persist model or thresholds:', e)

    with open(os.path.join(RESULTS_DIR, 'metrics_f1.json'), 'w', encoding='utf-8') as jf:
        json.dump({k:v for k,v in metrics.items() if k!='classification_report'}, jf, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS_DIR, 'metrics_recall_priority.json'), 'w', encoding='utf-8') as jf:
        json.dump({k:v for k,v in metrics_rec90.items() if k!='classification_report'}, jf, ensure_ascii=False, indent=2)

    test_out = female_df.iloc[test_idx].copy()
    test_out['y_true'] = y_test.values
    test_out['prob'] = prob_test
    test_out['pred_f1'] = y_pred
    test_out['pred_rec90'] = y_pred_rec90
    test_out.to_csv(os.path.join(RESULTS_DIR, 'test_predictions.csv'), index=False, encoding='utf-8-sig')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_recall', type=float, default=0.90, help='训练集敏感度优先的目标召回')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--results_dir', type=str, default=None, help='结果输出目录，默认使用脚本内置路径')
    parser.add_argument('--calibrate', action='store_true', help='是否进行概率校准（Platt/Isotonic）')
    parser.add_argument('--calib_size', type=float, default=0.2, help='从训练集中划分的校准集比例(0,1)')
    parser.add_argument('--calib_method', type=str, default='sigmoid', choices=['sigmoid', 'isotonic'], help='校准方法：sigmoid 或 isotonic')
    args = parser.parse_args()
    main(target_recall=args.target_recall, test_size=args.test_size, seed=args.seed,
         results_dir=args.results_dir, calibrate=args.calibrate, calib_size=args.calib_size, calib_method=args.calib_method)