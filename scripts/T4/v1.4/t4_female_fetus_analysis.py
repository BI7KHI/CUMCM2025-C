import os
import re
import json
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             classification_report, confusion_matrix, recall_score, 
                             precision_score, f1_score, make_scorer)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.mixture import GaussianMixture
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
import argparse
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ---------------------- 配置 ----------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SCRIPT_VERSION = 'v1.4'
DATA_A = os.path.join(BASE_DIR, 'Source_DATA', 'dataA.csv')
DATA_B = os.path.join(BASE_DIR, 'Source_DATA', 'dataB.csv')
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'T4', SCRIPT_VERSION)
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
RESULTS_DIR = DEFAULT_RESULTS_DIR

# ---------------------- 工具函数 ----------------------

def parse_gestational_age(s: str) -> Tuple[float, float]:
    """解析形如 '13w+6'、'13w' 的检测孕周，返回 (周_浮点, 天数)"""
    if pd.isna(s):
        return np.nan, np.nan
    s = str(s).strip()
    m = re.match(r"(\d+)w(?:\+(\d+))?", s)
    if m:
        w = int(m.group(1))
        d = int(m.group(2)) if m.group(2) else 0
        return w + d/7.0, w*7 + d
    try:
        v = float(s)
        return v, int(round(v*7))
    except Exception:
        return np.nan, np.nan

def read_and_unify(path: str) -> pd.DataFrame:
    """读取并统一数据格式"""
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    
    # 统一列名映射
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
    required_cols = ['Z13','Z18','Z21','ZX','ZY','Y浓度','X浓度','GC_total','GC13','GC18','GC21',
                    '比对比例','重复比例','过滤比例','BMI','原始读段数','唯一比对的读段数',
                    '孕妇代码','AB','胎儿是否健康','孕周_float','孕周_天数']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df

def improved_female_detection(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    改进的女胎检测方法，综合考虑Y染色体浓度、ZY值和X染色体特征
    """
    stats = {'method': 'improved_multi_criteria'}
    
    # 1. 基于Y染色体浓度的初步筛选
    y_conc = df['Y浓度'].dropna()
    zy_vals = df['ZY'].dropna()
    
    # 使用多个指标综合判断
    criteria_masks = []
    criteria_names = []
    
    # 准则1: Y染色体浓度极低或缺失
    if len(y_conc) > 0:
        y_threshold = np.percentile(y_conc, 30)  # 使用30分位数作为阈值
        mask1 = (df['Y浓度'].isna()) | (df['Y浓度'] <= y_threshold)
        criteria_masks.append(mask1)
        criteria_names.append(f'Y浓度≤{y_threshold:.4f}')
        stats['y_conc_threshold'] = float(y_threshold)
    
    # 准则2: ZY值显著偏低
    if len(zy_vals) > 0:
        zy_threshold = np.percentile(zy_vals, 25)  # 使用25分位数
        mask2 = (df['ZY'].isna()) | (df['ZY'] <= zy_threshold)
        criteria_masks.append(mask2)
        criteria_names.append(f'ZY≤{zy_threshold:.3f}')
        stats['zy_threshold'] = float(zy_threshold)
    
    # 准则3: X染色体特征正常化
    zx_vals = df['ZX'].dropna()
    if len(zx_vals) > 0:
        # X染色体Z值在正常范围内（不应过高或过低）
        zx_mean, zx_std = np.mean(zx_vals), np.std(zx_vals)
        mask3 = (df['ZX'] >= zx_mean - 2*zx_std) & (df['ZX'] <= zx_mean + 2*zx_std)
        criteria_masks.append(mask3)
        criteria_names.append(f'ZX正常范围')
        stats['zx_range'] = [float(zx_mean - 2*zx_std), float(zx_mean + 2*zx_std)]
    
    # 综合判断：满足多个准则
    if len(criteria_masks) >= 2:
        # 至少满足2个准则认为是女胎
        combined_mask = np.zeros(len(df), dtype=bool)
        for i in range(len(criteria_masks)):
            for j in range(i+1, len(criteria_masks)):
                combined_mask |= (criteria_masks[i] & criteria_masks[j])
        female_mask = combined_mask
    elif len(criteria_masks) == 1:
        female_mask = criteria_masks[0]
    else:
        # 兜底：全部视为女胎
        female_mask = np.ones(len(df), dtype=bool)
        stats['method'] = 'fallback_all_female'
    
    # 进一步验证：使用聚类方法
    if len(y_conc) >= 20:
        try:
            features_for_clustering = []
            if 'Y浓度' in df.columns:
                features_for_clustering.append('Y浓度')
            if 'ZY' in df.columns:
                features_for_clustering.append('ZY')
            
            if len(features_for_clustering) > 0:
                cluster_data = df[features_for_clustering].fillna(df[features_for_clustering].median())
                gmm = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(cluster_data)
                
                # 识别低Y浓度簇
                cluster_labels = gmm.predict(cluster_data)
                cluster_means = []
                for i in range(2):
                    cluster_mask = cluster_labels == i
                    if 'Y浓度' in features_for_clustering:
                        cluster_y_mean = df.loc[cluster_mask, 'Y浓度'].mean()
                        cluster_means.append(cluster_y_mean)
                
                if len(cluster_means) == 2:
                    female_cluster = np.argmin(cluster_means)
                    cluster_female_mask = cluster_labels == female_cluster
                    # 与前面的准则结合
                    female_mask = female_mask | cluster_female_mask
                    stats['clustering_validation'] = True
        except Exception as e:
            stats['clustering_error'] = str(e)
    
    filtered = df[female_mask].copy()
    stats.update({
        'total_samples': int(len(df)),
        'female_samples': int(len(filtered)),
        'female_ratio': float(len(filtered) / len(df)),
        'criteria_used': criteria_names
    })
    
    return filtered, stats

def advanced_feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    高级特征工程，重点关注染色体异常检测相关特征
    """
    # 标签：基于AB列的染色体异常
    ab = df['AB'].fillna('')
    y = ab.str.contains('T13|T18|T21', regex=True).astype(int)
    
    # 基础特征
    df['唯一比对率'] = df['唯一比对的读段数'] / (df['原始读段数'] + 1e-10)
    df['过滤率'] = df['过滤比例']
    df['测序深度'] = np.log1p(df['原始读段数'])
    
    # 染色体相关特征工程
    # 1. Z值组合特征
    df['Z13_abs'] = np.abs(df['Z13'])
    df['Z18_abs'] = np.abs(df['Z18']) 
    df['Z21_abs'] = np.abs(df['Z21'])
    df['ZX_abs'] = np.abs(df['ZX'])
    
    # Z值交互特征
    df['Z13_Z18_product'] = df['Z13'] * df['Z18']
    df['Z13_Z21_product'] = df['Z13'] * df['Z21']
    df['Z18_Z21_product'] = df['Z18'] * df['Z21']
    
    # Z值范数特征
    df['Z_norm_critical'] = np.sqrt(df['Z13']**2 + df['Z18']**2 + df['Z21']**2)
    df['Z_max_critical'] = np.maximum.reduce([df['Z13_abs'], df['Z18_abs'], df['Z21_abs']])
    
    # 2. GC含量特征工程
    df['GC_13_deviation'] = df['GC13'] - df['GC_total']
    df['GC_18_deviation'] = df['GC18'] - df['GC_total']
    df['GC_21_deviation'] = df['GC21'] - df['GC_total']
    
    # GC含量稳定性
    gc_cols = ['GC13', 'GC18', 'GC21']
    df['GC_variance'] = df[gc_cols].var(axis=1, skipna=True)
    df['GC_range'] = df[gc_cols].max(axis=1, skipna=True) - df[gc_cols].min(axis=1, skipna=True)
    
    # 3. 测序质量特征
    df['sequencing_quality'] = (df['比对比例'] * (1 - df['重复比例']) * df['唯一比对率'])
    df['data_quality_score'] = (df['sequencing_quality'] * np.log1p(df['唯一比对的读段数']))
    
    # 4. 临床特征
    df['BMI_category'] = pd.cut(df['BMI'], 
                              bins=[0, 18.5, 24, 28, 35, 100], 
                              labels=[0, 1, 2, 3, 4], 
                              include_lowest=True).astype(float)
    
    # 孕周相关特征
    df['孕周_squared'] = df['孕周_float'] ** 2
    df['孕周_BMI_interaction'] = df['孕周_float'] * df['BMI']
    
    # 5. 异常检测特征
    # 基于Z值的异常评分
    z_threshold = 2.5
    df['anomaly_13'] = (df['Z13_abs'] > z_threshold).astype(int)
    df['anomaly_18'] = (df['Z18_abs'] > z_threshold).astype(int)
    df['anomaly_21'] = (df['Z21_abs'] > z_threshold).astype(int)
    df['total_anomalies'] = df['anomaly_13'] + df['anomaly_18'] + df['anomaly_21']
    
    # 综合异常评分
    df['risk_score'] = (
        df['Z13_abs'] * 0.3 +  # T13权重
        df['Z18_abs'] * 0.4 +  # T18权重  
        df['Z21_abs'] * 0.3 +  # T21权重
        df['GC_variance'] * 0.1
    )
    
    # 选择最终特征
    feature_cols = [
        # 核心Z值特征
        'Z13', 'Z18', 'Z21', 'ZX',
        'Z13_abs', 'Z18_abs', 'Z21_abs', 'ZX_abs',
        
        # Z值交互特征
        'Z13_Z18_product', 'Z13_Z21_product', 'Z18_Z21_product',
        'Z_norm_critical', 'Z_max_critical',
        
        # GC含量特征
        'GC_total', 'GC13', 'GC18', 'GC21',
        'GC_13_deviation', 'GC_18_deviation', 'GC_21_deviation',
        'GC_variance', 'GC_range',
        
        # 测序质量特征
        '原始读段数', '测序深度', '比对比例', '重复比例', '唯一比对率', '过滤比例',
        'sequencing_quality', 'data_quality_score',
        
        # 临床特征
        'BMI', 'BMI_category', '孕周_float', '孕周_squared', '孕周_BMI_interaction',
        
        # 异常检测特征
        'anomaly_13', 'anomaly_18', 'anomaly_21', 'total_anomalies', 'risk_score'
    ]
    
    X = df[feature_cols].copy()
    
    # 数值化和清理
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 使用中位数填充缺失值
    X = X.fillna(X.median(numeric_only=True))
    
    return X, y, feature_cols

def enhanced_model_selection_and_training(X: pd.DataFrame, y: pd.Series, groups: pd.Series, 
                                        target_recall: float = 0.90, test_size: float = 0.2, 
                                        seed: int = 42) -> Tuple:
    """
    增强的模型选择和训练过程
    """
    # 数据划分
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"训练集样本数: {len(X_train)}, 正样本: {y_train.sum()}")
    print(f"测试集样本数: {len(X_test)}, 正样本: {y_test.sum()}")
    
    # 特征选择
    # 使用互信息和方差分析进行特征选择
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(20, X_train.shape[1]))
    selector_anova = SelectKBest(score_func=f_classif, k=min(20, X_train.shape[1]))
    
    X_train_mi = selector_mi.fit_transform(X_train, y_train)
    X_train_anova = selector_anova.fit_transform(X_train, y_train)
    
    selected_features_mi = X_train.columns[selector_mi.get_support()].tolist()
    selected_features_anova = X_train.columns[selector_anova.get_support()].tolist()
    
    # 合并特征选择结果
    selected_features = list(set(selected_features_mi + selected_features_anova))
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    print(f"选择的特征数量: {len(selected_features)}")
    print(f"选择的特征: {selected_features}")
    
    # 模型候选列表
    models = {
        'LogisticRegression': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=2000, random_state=seed, class_weight='balanced'))
            ]),
            'params': {
                'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'clf__penalty': ['l1', 'l2'],
                'clf__solver': ['liblinear']
            }
        },
        'RandomForest': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(random_state=seed, class_weight='balanced', n_jobs=-1))
            ]),
            'params': {
                'clf__n_estimators': [100, 200, 300],
                'clf__max_depth': [3, 5, 7, None],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4]
            }
        },
        'GradientBoosting': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(random_state=seed))
            ]),
            'params': {
                'clf__n_estimators': [100, 200],
                'clf__learning_rate': [0.01, 0.1, 0.2],
                'clf__max_depth': [3, 5, 7],
                'clf__subsample': [0.8, 1.0]
            }
        }
    }
    
    # 使用分层分组交叉验证
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    
    # 自定义评分函数：平衡F1和Recall
    def balanced_scorer(y_true, y_pred):
        f1 = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        return 0.6 * f1 + 0.4 * recall
    
    balanced_score = make_scorer(balanced_scorer)
    
    best_model = None
    best_score = -1
    best_model_name = ""
    model_results = {}
    
    # 模型选择和调优
    for name, model_config in models.items():
        print(f"\n训练模型: {name}")
        
        try:
            gs = GridSearchCV(
                model_config['model'], 
                model_config['params'],
                scoring=balanced_score,
                cv=cv,
                n_jobs=-1,
                refit=True
            )
            
            gs.fit(X_train_selected, y_train, groups=groups.iloc[train_idx])
            
            model_results[name] = {
                'best_score': gs.best_score_,
                'best_params': gs.best_params_,
                'model': gs.best_estimator_
            }
            
            print(f"{name} 最佳评分: {gs.best_score_:.4f}")
            print(f"{name} 最佳参数: {gs.best_params_}")
            
            if gs.best_score_ > best_score:
                best_score = gs.best_score_
                best_model = gs.best_estimator_
                best_model_name = name
                
        except Exception as e:
            print(f"训练 {name} 时出错: {e}")
            continue
    
    print(f"\n最佳模型: {best_model_name}, 评分: {best_score:.4f}")
    
    return (best_model, X_train_selected, X_test_selected, y_train, y_test, 
            selected_features, model_results, best_model_name)

def comprehensive_evaluation_and_interpretation(model, X_train, X_test, y_train, y_test, 
                                               feature_names, target_recall=0.90):
    """
    全面的模型评估和可解释性分析
    """
    # 预测概率
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]
    
    # 阈值优化
    prec, rec, thr = precision_recall_curve(y_train, prob_train)
    prec_aligned = prec[:-1]
    rec_aligned = rec[:-1]
    f1_aligned = 2 * prec_aligned * rec_aligned / (prec_aligned + rec_aligned + 1e-12)
    
    # F1最优阈值
    if len(f1_aligned) > 0:
        best_f1_idx = np.argmax(f1_aligned)
        best_f1_threshold = thr[best_f1_idx] if len(thr) > 0 else 0.5
    else:
        best_f1_threshold = 0.5
    
    # 召回优先阈值
    recall_candidates = np.where(rec_aligned >= target_recall)[0]
    if len(recall_candidates) > 0 and len(thr) > 0:
        best_recall_idx = recall_candidates[np.argmax(f1_aligned[recall_candidates])]
        recall_threshold = thr[best_recall_idx]
    else:
        recall_threshold = best_f1_threshold
    
    # 测试集预测
    y_pred_f1 = (prob_test >= best_f1_threshold).astype(int)
    y_pred_recall = (prob_test >= recall_threshold).astype(int)
    
    # 计算指标
    metrics_f1 = {
        'threshold': float(best_f1_threshold),
        'precision': float(precision_score(y_test, y_pred_f1, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_f1, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_f1, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, prob_test)) if len(np.unique(y_test)) > 1 else None
    }
    
    metrics_recall = {
        'threshold': float(recall_threshold),
        'precision': float(precision_score(y_test, y_pred_recall, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_recall, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_recall, zero_division=0)),
        'roc_auc': metrics_f1['roc_auc']
    }
    
    # 特征重要性分析
    feature_importance = {}
    
    try:
        # 1. 基于模型的特征重要性
        if hasattr(model.named_steps['clf'], 'feature_importances_'):
            model_importance = model.named_steps['clf'].feature_importances_
            feature_importance['model_based'] = dict(zip(feature_names, model_importance))
        elif hasattr(model.named_steps['clf'], 'coef_'):
            model_importance = np.abs(model.named_steps['clf'].coef_[0])
            feature_importance['model_based'] = dict(zip(feature_names, model_importance))
        
        # 2. 排列重要性
        perm_importance = permutation_importance(model, X_test, y_test, 
                                               n_repeats=5, random_state=42, 
                                               scoring='f1')
        feature_importance['permutation'] = dict(zip(feature_names, perm_importance.importances_mean))
        
    except Exception as e:
        print(f"特征重要性计算错误: {e}")
    
    return (metrics_f1, metrics_recall, y_pred_f1, y_pred_recall, prob_test, 
            best_f1_threshold, recall_threshold, feature_importance, prec, rec, thr)

def create_comprehensive_visualizations(results_dict, feature_importance, feature_names):
    """
    创建全面的可视化图表
    """
    metrics_f1, metrics_recall, y_pred_f1, y_pred_recall, prob_test = results_dict['metrics'][:5]
    y_test, prec, rec, thr = results_dict['data'][:4]
    
    # 1. PR曲线和ROC曲线
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # PR曲线
    ax1.plot(rec, prec, 'b-', linewidth=2, label='PR Curve')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ROC曲线
    if metrics_f1['roc_auc'] is not None:
        fpr, tpr, _ = roc_curve(y_test, prob_test)
        ax2.plot(fpr, tpr, 'r-', linewidth=2, label=f'ROC (AUC={metrics_f1["roc_auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 混淆矩阵 - F1优化
    cm1 = confusion_matrix(y_test, y_pred_f1)
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
    ax3.set_title(f'Confusion Matrix (F1-Optimal)\nThreshold: {metrics_f1["threshold"]:.3f}')
    
    # 混淆矩阵 - 召回优先
    cm2 = confusion_matrix(y_test, y_pred_recall)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=ax4,
                xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
    ax4.set_title(f'Confusion Matrix (Recall-Priority)\nThreshold: {metrics_recall["threshold"]:.3f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comprehensive_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 特征重要性可视化
    if feature_importance:
        fig, axes = plt.subplots(1, len(feature_importance), figsize=(15, 6))
        if len(feature_importance) == 1:
            axes = [axes]
        
        for idx, (method, importance) in enumerate(feature_importance.items()):
            # 选择top 15特征
            sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            features, scores = zip(*sorted_features)
            
            y_pos = np.arange(len(features))
            axes[idx].barh(y_pos, scores)
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(features)
            axes[idx].set_title(f'Feature Importance ({method})')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

def save_comprehensive_results(all_results, feature_importance, selected_features, model_results):
    """
    保存全面的结果文件
    """
    # 保存主要指标
    with open(os.path.join(RESULTS_DIR, 'comprehensive_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'version': SCRIPT_VERSION,
            'f1_optimized': all_results['metrics_f1'],
            'recall_optimized': all_results['metrics_recall'],
            'feature_count': len(selected_features),
            'selected_features': selected_features
        }, f, ensure_ascii=False, indent=2)
    
    # 保存特征重要性
    if feature_importance:
        with open(os.path.join(RESULTS_DIR, 'feature_importance.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'version': SCRIPT_VERSION,
                'importance_scores': feature_importance
            }, f, ensure_ascii=False, indent=2)
    
    # 保存模型对比结果
    model_comparison = {}
    for name, result in model_results.items():
        model_comparison[name] = {
            'best_score': result['best_score'],
            'best_params': result['best_params']
        }
    
    with open(os.path.join(RESULTS_DIR, 'model_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'version': SCRIPT_VERSION,
            'model_results': model_comparison
        }, f, ensure_ascii=False, indent=2)

def main(target_recall: float = 0.90, test_size: float = 0.2, seed: int = 42,
         results_dir: Optional[str] = None):
    """主函数"""
    np.random.seed(seed)
    if results_dir is not None:
        global RESULTS_DIR
        RESULTS_DIR = results_dir
        os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"=== 女胎异常检测分析 {SCRIPT_VERSION} ===")
    
    # 1. 读取和合并数据
    print("1. 读取数据...")
    dfa = read_and_unify(DATA_A)
    dfb = read_and_unify(DATA_B)
    raw = pd.concat([dfa, dfb], ignore_index=True)
    print(f"原始数据量: {len(raw)}")
    
    # 2. 改进的女胎筛选
    print("2. 女胎样本筛选...")
    female_df, female_stats = improved_female_detection(raw)
    print(f"女胎样本数: {len(female_df)} ({female_stats['female_ratio']:.2%})")
    
    # 保存女胎筛选统计
    with open(os.path.join(RESULTS_DIR, 'female_selection_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(female_stats, f, ensure_ascii=False, indent=2)
    
    # 3. 高级特征工程
    print("3. 特征工程...")
    X, y, feature_names = advanced_feature_engineering(female_df)
    groups = female_df['孕妇代码'].astype(str).fillna('NA')
    print(f"特征数量: {len(feature_names)}")
    print(f"标签分布: 正常={len(y)-y.sum()}, 异常={y.sum()}")
    
    # 4. 模型选择和训练
    print("4. 模型训练和选择...")
    (best_model, X_train, X_test, y_train, y_test, 
     selected_features, model_results, best_model_name) = enhanced_model_selection_and_training(
        X, y, groups, target_recall, test_size, seed)
    
    # 5. 全面评估和解释
    print("5. 模型评估和解释...")
    (metrics_f1, metrics_recall, y_pred_f1, y_pred_recall, prob_test, 
     best_f1_threshold, recall_threshold, feature_importance, prec, rec, thr) = comprehensive_evaluation_and_interpretation(
        best_model, X_train, X_test, y_train, y_test, selected_features, target_recall)
    
    # 打印主要结果
    print(f"\n=== 最终结果 ===")
    print(f"最佳模型: {best_model_name}")
    print(f"F1优化 - 精确率: {metrics_f1['precision']:.3f}, 召回率: {metrics_f1['recall']:.3f}, F1: {metrics_f1['f1']:.3f}")
    print(f"召回优先 - 精确率: {metrics_recall['precision']:.3f}, 召回率: {metrics_recall['recall']:.3f}, F1: {metrics_recall['f1']:.3f}")
    if metrics_f1['roc_auc'] is not None:
        print(f"ROC AUC: {metrics_f1['roc_auc']:.3f}")
    
    # 6. 生成可视化
    print("6. 生成可视化图表...")
    results_for_viz = {
        'metrics': (metrics_f1, metrics_recall, y_pred_f1, y_pred_recall, prob_test),
        'data': (y_test, prec, rec, thr)
    }
    create_comprehensive_visualizations(results_for_viz, feature_importance, selected_features)
    
    # 7. 保存结果
    print("7. 保存结果文件...")
    all_results = {
        'metrics_f1': metrics_f1,
        'metrics_recall': metrics_recall,
        'female_stats': female_stats,
        'best_model_name': best_model_name
    }
    save_comprehensive_results(all_results, feature_importance, selected_features, model_results)
    
    # 保存测试集预测结果
    test_indices = X_test.index
    test_results = female_df.loc[test_indices].copy()
    test_results['y_true'] = y_test.values
    test_results['prob_score'] = prob_test
    test_results['pred_f1_optimal'] = y_pred_f1
    test_results['pred_recall_priority'] = y_pred_recall
    test_results.to_csv(os.path.join(RESULTS_DIR, 'test_predictions_detailed.csv'), 
                       index=False, encoding='utf-8-sig')
    
    # 保存模型
    try:
        import joblib
        joblib.dump(best_model, os.path.join(RESULTS_DIR, 'best_model.joblib'))
        print("模型已保存")
    except Exception as e:
        print(f"模型保存失败: {e}")
    
    print(f"\n所有结果已保存到: {RESULTS_DIR}")
    return best_model, all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='女胎异常检测分析 v1.4')
    parser.add_argument('--target_recall', type=float, default=0.90, 
                       help='目标召回率')
    parser.add_argument('--test_size', type=float, default=0.2, 
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子')
    parser.add_argument('--results_dir', type=str, default=None, 
                       help='结果输出目录')
    
    args = parser.parse_args()
    main(target_recall=args.target_recall, test_size=args.test_size, 
         seed=args.seed, results_dir=args.results_dir)
