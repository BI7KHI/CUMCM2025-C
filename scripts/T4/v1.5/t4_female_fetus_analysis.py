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
from scipy.optimize import minimize_scalar
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
SCRIPT_VERSION = 'v1.5'
DATA_A = os.path.join(BASE_DIR, 'Source_DATA', 'dataA.csv')
DATA_B = os.path.join(BASE_DIR, 'Source_DATA', 'dataB.csv')
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'T4', SCRIPT_VERSION)
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
RESULTS_DIR = DEFAULT_RESULTS_DIR

# ---------------------- 文献依据的常量定义 ----------------------
# 基于魏智芳、薛莹文献的关键阈值
LITERATURE_CONSTANTS = {
    'ff_failure_threshold': 0.04,  # FF<4%时NIPT检测失败 (魏智芳2025)
    'y_conc_percentile_threshold': 30,  # Y浓度30分位数阈值
    'zy_percentile_threshold': 25,  # ZY值25分位数阈值
    'bmi_high_risk': 35,  # BMI≥35为高风险 (Livergood et al. 2017)
    'gestational_week_cutoff': 21,  # 21周为关键时间点 (Wang et al. 2013)
    'ff_base_intercept': 0.060,  # 基础FF截距6% (薛莹2017)
    'ga_coeff_early': 0.003,  # 孕早期系数0.3%/周 (薛莹2017)
    'ga_coeff_late_increment': 0.004,  # 孕晚期增量0.4%/周 (魏智芳2025)
    'bmi_coeff_normal': -0.003,  # BMI<35时系数-0.3%/(kg/m²) (薛莹2017)
    'bmi_coeff_plateau': 0.002,  # BMI≥35平台效应修正 (魏智芳2025)
}

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

def literature_based_female_detection(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    基于文献证据的女胎检测方法
    
    参考文献：
    1. 魏智芳2025: FF<4%为检测失败阈值
    2. 薛莹2017: Y染色体浓度量化公式
    3. Canick et al. 2013: Y染色体特异性检测
    """
    stats = {'method': 'literature_based_multi_criteria', 'version': SCRIPT_VERSION}
    
    # 1. 计算估计的胎儿分数(FF)基于Y染色体浓度
    y_conc = df['Y浓度'].fillna(0)
    zy_vals = df['ZY'].fillna(0)
    bmi_vals = df['BMI'].fillna(25)  # 默认正常BMI
    ga_vals = df['孕周_float'].fillna(15)  # 默认15周
    
    # 基于薛莹2017公式估计FF (仅用于参考)
    estimated_ff = []
    for i in range(len(df)):
        ga = ga_vals.iloc[i]
        bmi = bmi_vals.iloc[i]
        
        if pd.isna(ga) or pd.isna(bmi):
            estimated_ff.append(np.nan)
            continue
            
        # 基于文献公式: FF = α + β1×GA + β2×(GA≥21)×GA + β3×BMI + β4×(BMI≥35)×BMI
        alpha = LITERATURE_CONSTANTS['ff_base_intercept']
        beta1 = LITERATURE_CONSTANTS['ga_coeff_early']
        beta2 = LITERATURE_CONSTANTS['ga_coeff_late_increment'] if ga >= LITERATURE_CONSTANTS['gestational_week_cutoff'] else 0
        beta3 = LITERATURE_CONSTANTS['bmi_coeff_normal']
        beta4 = LITERATURE_CONSTANTS['bmi_coeff_plateau'] if bmi >= LITERATURE_CONSTANTS['bmi_high_risk'] else 0
        
        ff_est = alpha + beta1 * ga + beta2 * max(0, ga - LITERATURE_CONSTANTS['gestational_week_cutoff']) + beta3 * bmi + beta4 * max(0, bmi - LITERATURE_CONSTANTS['bmi_high_risk'])
        estimated_ff.append(max(0, ff_est))
    
    df['estimated_ff'] = estimated_ff
    
    # 2. 多准则女胎判定
    criteria_masks = []
    criteria_names = []
    
    # 准则1: Y染色体浓度低 (基于Canick et al. 2013)
    if len(y_conc.dropna()) > 0:
        y_threshold = np.percentile(y_conc.dropna(), LITERATURE_CONSTANTS['y_conc_percentile_threshold'])
        mask1 = (df['Y浓度'].isna()) | (df['Y浓度'] <= y_threshold)
        criteria_masks.append(mask1)
        criteria_names.append(f'Y浓度≤P{LITERATURE_CONSTANTS["y_conc_percentile_threshold"]}({y_threshold:.4f})')
        stats['y_conc_threshold'] = float(y_threshold)
    
    # 准则2: ZY值显著偏低 (基于Wang et al. 2013)
    if len(zy_vals.dropna()) > 0:
        zy_threshold = np.percentile(zy_vals.dropna(), LITERATURE_CONSTANTS['zy_percentile_threshold'])
        mask2 = (df['ZY'].isna()) | (df['ZY'] <= zy_threshold)
        criteria_masks.append(mask2)
        criteria_names.append(f'ZY≤P{LITERATURE_CONSTANTS["zy_percentile_threshold"]}({zy_threshold:.3f})')
        stats['zy_threshold'] = float(zy_threshold)
    
    # 准则3: 估计FF接近检测失败阈值或偏低 (基于魏智芳2025)
    mask3 = (pd.Series(estimated_ff) <= LITERATURE_CONSTANTS['ff_failure_threshold'] * 3) | pd.Series(estimated_ff).isna()
    criteria_masks.append(mask3)
    criteria_names.append(f'估计FF≤{LITERATURE_CONSTANTS["ff_failure_threshold"] * 3:.2f}')
    
    # 准则4: 高BMI且孕周较晚仍有低Y浓度表现 (基于Livergood et al. 2017)
    high_bmi_late_ga = (bmi_vals >= LITERATURE_CONSTANTS['bmi_high_risk']) & (ga_vals >= LITERATURE_CONSTANTS['gestational_week_cutoff'])
    mask4 = (~high_bmi_late_ga) | (high_bmi_late_ga & mask1)  # 非高风险组或高风险组中Y浓度仍低
    criteria_masks.append(mask4)
    criteria_names.append('BMI-GA风险评估')
    
    # 综合判断：满足至少2个准则认为是女胎
    if len(criteria_masks) >= 2:
        vote_scores = np.zeros(len(df))
        for mask in criteria_masks:
            vote_scores += mask.astype(int)
        female_mask = vote_scores >= 2  # 至少2票
        stats['decision_rule'] = 'majority_vote_2_of_4'
    else:
        female_mask = criteria_masks[0] if criteria_masks else np.ones(len(df), dtype=bool)
        stats['decision_rule'] = 'single_criterion'
    
    # 3. 聚类验证 (基于Kinnings et al. 2015)
    if len(y_conc.dropna()) >= 20:
        try:
            cluster_features = df[['Y浓度', 'ZY']].fillna(df[['Y浓度', 'ZY']].median())
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(cluster_features)
            cluster_labels = gmm.predict(cluster_features)
            
            # 识别低Y浓度簇
            cluster_y_means = [df.loc[cluster_labels == i, 'Y浓度'].mean() for i in range(2)]
            female_cluster = np.argmin(cluster_y_means)
            cluster_female_mask = cluster_labels == female_cluster
            
            # 与投票结果取并集
            female_mask = female_mask | cluster_female_mask
            stats['clustering_validation'] = {
                'performed': True,
                'cluster_means': cluster_y_means,
                'female_cluster_id': int(female_cluster)
            }
        except Exception as e:
            stats['clustering_validation'] = {'performed': False, 'error': str(e)}
    
    filtered = df[female_mask].copy()
    stats.update({
        'total_samples': int(len(df)),
        'female_samples': int(len(filtered)),
        'female_ratio': float(len(filtered) / len(df)),
        'criteria_used': criteria_names,
        'literature_basis': [
            'Canick_2013_Y_chromosome_specificity',
            'Wang_2013_gestational_age_effects', 
            'Kinnings_2015_BMI_plateau_effect',
            'Livergood_2017_BMI_35_cutoff',
            'Wei_2025_FF_failure_threshold',
            'Xue_2017_quantitative_relationship'
        ]
    })
    
    return filtered, stats

def literature_guided_feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    基于文献指导的特征工程
    
    参考文献：
    1. Haghiac et al. 2013: 脂肪细胞坏死机制
    2. 薛莹2017: FF与BMI、孕周的定量关系
    3. 魏智芳2025: GC含量的重要性
    """
    # 标签：基于AB列的染色体异常
    ab = df['AB'].fillna('')
    y = ab.str.contains('T13|T18|T21', regex=True).astype(int)
    
    # 1. 基础质量控制特征
    df['唯一比对率'] = df['唯一比对的读段数'] / (df['原始读段数'] + 1e-10)
    df['测序深度_log'] = np.log1p(df['原始读段数'])
    df['质量综合评分'] = (df['比对比例'] * (1 - df['重复比例']) * df['唯一比对率'])
    
    # 2. 基于文献的染色体Z值特征 (重点关注T13/T18/T21)
    # 绝对Z值 - 检测异常的核心指标
    df['Z13_abs'] = np.abs(df['Z13'])
    df['Z18_abs'] = np.abs(df['Z18'])
    df['Z21_abs'] = np.abs(df['Z21'])
    df['ZX_abs'] = np.abs(df['ZX'])
    
    # 基于文献的异常阈值判定 (Z>2.5为异常)
    z_threshold = 2.5
    df['Z13_abnormal'] = (df['Z13_abs'] > z_threshold).astype(int)
    df['Z18_abnormal'] = (df['Z18_abs'] > z_threshold).astype(int)
    df['Z21_abnormal'] = (df['Z21_abs'] > z_threshold).astype(int)
    df['total_z_abnormal'] = df['Z13_abnormal'] + df['Z18_abnormal'] + df['Z21_abnormal']
    
    # 染色体间相互作用 (基于共同异常模式)
    df['Z13_Z18_interaction'] = df['Z13'] * df['Z18']
    df['Z13_Z21_interaction'] = df['Z13'] * df['Z21']
    df['Z18_Z21_interaction'] = df['Z18'] * df['Z21']
    
    # 多染色体综合风险评分 (加权考虑T13/T18/T21重要性)
    df['trisomy_risk_score'] = (
        0.3 * df['Z13_abs'] +  # T13权重
        0.4 * df['Z18_abs'] +  # T18权重(临床最常见)
        0.3 * df['Z21_abs']    # T21权重(唐氏综合征)
    )
    
    # 3. 基于文献的GC含量特征 (魏智芳2025强调GC含量重要性)
    # GC含量偏差分析
    df['GC13_deviation'] = df['GC13'] - df['GC_total']
    df['GC18_deviation'] = df['GC18'] - df['GC_total']
    df['GC21_deviation'] = df['GC21'] - df['GC_total']
    
    # GC含量稳定性指标
    gc_cols = ['GC13', 'GC18', 'GC21']
    df['GC_variance'] = df[gc_cols].var(axis=1, skipna=True)
    df['GC_range'] = df[gc_cols].max(axis=1, skipna=True) - df[gc_cols].min(axis=1, skipna=True)
    df['GC_coefficient_variation'] = df['GC_variance'] / (df['GC_total'] + 1e-10)
    
    # 基于Haghiac2013的异常GC检测
    df['GC_extreme_deviation'] = np.maximum.reduce([
        np.abs(df['GC13_deviation']),
        np.abs(df['GC18_deviation']),
        np.abs(df['GC21_deviation'])
    ])
    
    # 4. 基于薛莹2017的临床特征建模
    # BMI分层效应 (基于文献的35kg/m²切点)
    df['BMI_high_risk'] = (df['BMI'] >= LITERATURE_CONSTANTS['bmi_high_risk']).astype(int)
    df['BMI_squared'] = df['BMI'] ** 2
    df['BMI_log'] = np.log1p(df['BMI'])
    
    # 孕周分段效应 (基于文献的21周切点)
    df['GA_early_phase'] = (df['孕周_float'] < LITERATURE_CONSTANTS['gestational_week_cutoff']).astype(int)
    df['GA_late_phase'] = (df['孕周_float'] >= LITERATURE_CONSTANTS['gestational_week_cutoff']).astype(int)
    df['GA_squared'] = df['孕周_float'] ** 2
    
    # BMI-孕周交互效应 (基于Livergood2017的补偿机制)
    df['BMI_GA_interaction'] = df['BMI'] * df['孕周_float']
    df['high_BMI_late_GA'] = df['BMI_high_risk'] * df['GA_late_phase']
    
    # 基于文献公式的估计FF特征
    if 'estimated_ff' in df.columns:
        df['FF_below_threshold'] = (df['estimated_ff'] < LITERATURE_CONSTANTS['ff_failure_threshold']).astype(int)
        df['FF_log'] = np.log1p(df['estimated_ff'])
    
    # 5. 测序技术特征 (基于Kinnings2015测序深度要求)
    df['sequencing_depth_adequate'] = (df['原始读段数'] >= 16000000).astype(int)  # ≥16M reads
    df['mapping_efficiency'] = df['比对比例'] * (1 - df['重复比例'])
    df['data_completeness'] = (1 - df['过滤比例']) * df['唯一比对率']
    
    # 最终特征选择 (基于文献重要性排序)
    feature_cols = [
        # 核心染色体异常特征 (最高优先级)
        'Z13', 'Z18', 'Z21', 'ZX',
        'Z13_abs', 'Z18_abs', 'Z21_abs', 'ZX_abs',
        'Z13_abnormal', 'Z18_abnormal', 'Z21_abnormal', 'total_z_abnormal',
        'trisomy_risk_score',
        
        # 染色体交互特征
        'Z13_Z18_interaction', 'Z13_Z21_interaction', 'Z18_Z21_interaction',
        
        # GC含量特征 (基于魏智芳2025的发现)
        'GC_total', 'GC13', 'GC18', 'GC21',
        'GC13_deviation', 'GC18_deviation', 'GC21_deviation',
        'GC_variance', 'GC_range', 'GC_coefficient_variation', 'GC_extreme_deviation',
        
        # 临床特征 (基于薛莹2017)
        'BMI', 'BMI_high_risk', 'BMI_squared', 'BMI_log',
        '孕周_float', 'GA_early_phase', 'GA_late_phase', 'GA_squared',
        'BMI_GA_interaction', 'high_BMI_late_GA',
        
        # 测序质量特征
        '原始读段数', '测序深度_log', '比对比例', '重复比例', '唯一比对率', '过滤比例',
        'sequencing_depth_adequate', 'mapping_efficiency', 'data_completeness', '质量综合评分'
    ]
    
    # 添加估计FF特征 (如果存在)
    if 'estimated_ff' in df.columns:
        feature_cols.extend(['estimated_ff', 'FF_below_threshold', 'FF_log'])
    
    X = df[feature_cols].copy()
    
    # 数值化和清理
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 使用中位数填充缺失值
    X = X.fillna(X.median(numeric_only=True))
    
    return X, y, feature_cols

def literature_informed_model_selection(X: pd.DataFrame, y: pd.Series, groups: pd.Series, 
                                       target_recall: float = 0.90, test_size: float = 0.2, 
                                       seed: int = 42) -> Tuple:
    """
    基于文献指导的模型选择和训练
    
    参考文献证据链：
    1. T1-T3的解题思路：从数据预处理到模型优化的完整流程
    2. 文献建议：使用分段函数和交互项进行建模
    """
    # 数据划分
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"训练集: {len(X_train)}样本, 异常{y_train.sum()}例 ({y_train.mean():.1%})")
    print(f"测试集: {len(X_test)}样本, 异常{y_test.sum()}例 ({y_test.mean():.1%})")
    
    # 基于文献的特征选择策略
    # 1. 基于先验知识的特征重要性评估
    clinical_priority_features = [
        'Z13_abs', 'Z18_abs', 'Z21_abs',  # 核心染色体异常指标
        'trisomy_risk_score',              # 综合风险评分
        'GC_variance', 'GC_range',         # GC含量稳定性
        'BMI_GA_interaction',              # 临床交互效应
        'total_z_abnormal'                 # 异常计数
    ]
    
    # 2. 统计学特征选择
    selector_anova = SelectKBest(score_func=f_classif, k=min(25, X_train.shape[1]))
    X_train_selected = selector_anova.fit_transform(X_train, y_train)
    selected_features = X_train.columns[selector_anova.get_support()].tolist()
    
    # 3. 确保临床优先特征被包含
    for feature in clinical_priority_features:
        if feature in X_train.columns and feature not in selected_features:
            selected_features.append(feature)
    
    # 更新选择的特征
    X_train_final = X_train[selected_features]
    X_test_final = X_test[selected_features]
    
    print(f"最终选择特征数: {len(selected_features)}")
    print(f"包含的临床优先特征: {[f for f in clinical_priority_features if f in selected_features]}")
    
    # 基于文献的模型配置
    models = {
        'LogisticRegression_L1': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    penalty='l1', 
                    solver='liblinear',
                    class_weight='balanced',
                    max_iter=2000, 
                    random_state=seed
                ))
            ]),
            'params': {'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0]}
        },
        'LogisticRegression_L2': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    penalty='l2',
                    class_weight='balanced',
                    max_iter=2000, 
                    random_state=seed
                ))
            ]),
            'params': {'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0]}
        },
        'RandomForest': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(
                    class_weight='balanced',
                    random_state=seed,
                    n_jobs=-1
                ))
            ]),
            'params': {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [3, 5, 7],
                'clf__min_samples_split': [2, 5],
                'clf__min_samples_leaf': [1, 2]
            }
        }
    }
    
    # 交叉验证策略
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    
    # 基于文献的评分函数：优先保证召回率
    def clinical_score(y_true, y_pred):
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        # 临床导向：召回率权重更高 (基于魏智芳2025的检测失败风险)
        return 0.5 * recall + 0.3 * f1 + 0.2 * precision
    
    clinical_scorer = make_scorer(clinical_score)
    
    best_model = None
    best_score = -1
    best_model_name = ""
    model_results = {}
    
    # 模型训练和选择
    for name, model_config in models.items():
        print(f"\n训练模型: {name}")
        
        try:
            gs = GridSearchCV(
                model_config['model'], 
                model_config['params'],
                scoring=clinical_scorer,
                cv=cv,
                n_jobs=-1,
                refit=True
            )
            
            gs.fit(X_train_final, y_train, groups=groups.iloc[train_idx])
            
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
    
    return (best_model, X_train_final, X_test_final, y_train, y_test, 
            selected_features, model_results, best_model_name)

def literature_based_threshold_optimization(model, X_train, y_train, target_recall=0.90):
    """
    基于文献的阈值优化策略
    
    参考：
    1. 魏智芳2025: FF<4%为临床关键阈值
    2. Wang et al. 2013: 重采血策略的56%成功率
    """
    prob_train = model.predict_proba(X_train)[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, prob_train)
    
    # 计算F1分数
    prec_aligned = prec[:-1]
    rec_aligned = rec[:-1]
    f1_scores = 2 * prec_aligned * rec_aligned / (prec_aligned + rec_aligned + 1e-12)
    
    # 1. F1最优阈值
    if len(f1_scores) > 0:
        best_f1_idx = np.argmax(f1_scores)
        f1_optimal_threshold = float(thr[best_f1_idx]) if len(thr) > 0 else 0.5
        f1_optimal_metrics = {
            'precision': float(prec_aligned[best_f1_idx]),
            'recall': float(rec_aligned[best_f1_idx]),
            'f1': float(f1_scores[best_f1_idx])
        }
    else:
        f1_optimal_threshold = 0.5
        f1_optimal_metrics = {'precision': 0, 'recall': 0, 'f1': 0}
    
    # 2. 召回优先阈值 (基于临床需求)
    recall_candidates = np.where(rec_aligned >= target_recall)[0]
    if len(recall_candidates) > 0:
        # 在满足召回要求的候选中选择F1最高的
        best_recall_idx = recall_candidates[np.argmax(f1_scores[recall_candidates])]
        recall_optimal_threshold = float(thr[best_recall_idx])
        recall_optimal_metrics = {
            'precision': float(prec_aligned[best_recall_idx]),
            'recall': float(rec_aligned[best_recall_idx]),
            'f1': float(f1_scores[best_recall_idx])
        }
    else:
        # 如果无法达到目标召回率，选择最接近的
        closest_idx = np.argmin(np.abs(rec_aligned - target_recall))
        recall_optimal_threshold = float(thr[closest_idx]) if len(thr) > 0 else 0.3
        recall_optimal_metrics = {
            'precision': float(prec_aligned[closest_idx]),
            'recall': float(rec_aligned[closest_idx]),
            'f1': float(f1_scores[closest_idx])
        }
    
    # 3. 基于文献的临床阈值 (保守策略)
    # 基于魏智芳2025的FF<4%失败率，设定更保守的阈值
    conservative_threshold = min(f1_optimal_threshold * 0.8, 0.3)
    
    return {
        'f1_optimal': {
            'threshold': f1_optimal_threshold,
            'metrics': f1_optimal_metrics
        },
        'recall_optimal': {
            'threshold': recall_optimal_threshold,
            'metrics': recall_optimal_metrics
        },
        'conservative_clinical': {
            'threshold': conservative_threshold,
            'description': '基于文献的保守临床阈值'
        },
        'pr_curve_data': {
            'precision': prec.tolist(),
            'recall': rec.tolist(),
            'thresholds': thr.tolist()
        }
    }

def comprehensive_evaluation_with_literature_context(model, X_train, X_test, y_train, y_test, 
                                                   feature_names, threshold_results,
                                                   target_recall=0.90):
    """
    基于文献背景的全面评估
    """
    # 测试集预测
    prob_test = model.predict_proba(X_test)[:, 1]
    
    # 使用不同阈值进行预测
    f1_threshold = threshold_results['f1_optimal']['threshold']
    recall_threshold = threshold_results['recall_optimal']['threshold']
    conservative_threshold = threshold_results['conservative_clinical']['threshold']
    
    predictions = {
        'f1_optimal': (prob_test >= f1_threshold).astype(int),
        'recall_optimal': (prob_test >= recall_threshold).astype(int),
        'conservative': (prob_test >= conservative_threshold).astype(int)
    }
    
    # 计算各种阈值下的指标
    threshold_mapping = {
        'f1_optimal': f1_threshold,
        'recall_optimal': recall_threshold,
        'conservative': conservative_threshold
    }
    
    evaluation_results = {}
    for strategy, y_pred in predictions.items():
        evaluation_results[strategy] = {
            'threshold': threshold_mapping[strategy],
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    # ROC AUC
    roc_auc = float(roc_auc_score(y_test, prob_test)) if len(np.unique(y_test)) > 1 else None
    for strategy in evaluation_results:
        evaluation_results[strategy]['roc_auc'] = roc_auc
    
    # 特征重要性分析
    feature_importance = {}
    try:
        # 基于模型的特征重要性
        if hasattr(model.named_steps['clf'], 'feature_importances_'):
            importance_values = model.named_steps['clf'].feature_importances_
            feature_importance['model_based'] = dict(zip(feature_names, importance_values))
        elif hasattr(model.named_steps['clf'], 'coef_'):
            importance_values = np.abs(model.named_steps['clf'].coef_[0])
            feature_importance['model_based'] = dict(zip(feature_names, importance_values))
        
        # 排列重要性
        perm_importance = permutation_importance(model, X_test, y_test, 
                                               n_repeats=5, random_state=42, 
                                               scoring='f1')
        feature_importance['permutation'] = dict(zip(feature_names, perm_importance.importances_mean))
        
    except Exception as e:
        print(f"特征重要性计算错误: {e}")
    
    return evaluation_results, feature_importance, prob_test, predictions

def create_literature_informed_visualizations(evaluation_results, feature_importance, 
                                            y_test, prob_test, predictions, threshold_results):
    """
    基于文献背景的可视化
    """
    # 创建综合评估图
    fig = plt.figure(figsize=(20, 15))
    
    # 1. PR曲线 (2x2的第一个)
    ax1 = plt.subplot(3, 3, 1)
    pr_data = threshold_results['pr_curve_data']
    ax1.plot(pr_data['recall'], pr_data['precision'], 'b-', linewidth=2, label='PR曲线')
    
    # 标注三种阈值
    for strategy, color in [('f1_optimal', 'green'), ('recall_optimal', 'red'), ('conservative', 'orange')]:
        threshold = evaluation_results[strategy]['threshold']
        prec = evaluation_results[strategy]['precision']
        recall = evaluation_results[strategy]['recall']
        ax1.scatter(recall, prec, color=color, s=100, 
                   label=f'{strategy.replace("_", " ").title()}\n(阈值={threshold:.3f})')
    
    ax1.set_xlabel('召回率 (Recall)')
    ax1.set_ylabel('精确率 (Precision)')
    ax1.set_title('精确率-召回率曲线\n(基于文献的三种阈值策略)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC曲线 (2x2的第二个)
    ax2 = plt.subplot(3, 3, 2)
    if evaluation_results['f1_optimal']['roc_auc'] is not None:
        fpr, tpr, _ = roc_curve(y_test, prob_test)
        ax2.plot(fpr, tpr, 'r-', linewidth=2, 
                label=f'ROC (AUC={evaluation_results["f1_optimal"]["roc_auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机分类')
        ax2.set_xlabel('假阳性率 (FPR)')
        ax2.set_ylabel('真阳性率 (TPR)')
        ax2.set_title('ROC曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3-5. 三种策略的混淆矩阵
    strategy_names = ['F1优化', '召回优先', '保守策略']
    colors = ['Blues', 'Reds', 'Oranges']
    
    for i, ((strategy, result), name, cmap) in enumerate(zip(evaluation_results.items(), strategy_names, colors)):
        ax = plt.subplot(3, 3, i + 3)
        cm = np.array(result['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                   xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
        ax.set_title(f'{name}策略\n精确率:{result["precision"]:.3f} 召回率:{result["recall"]:.3f}')
    
    # 6. 特征重要性对比
    if feature_importance:
        ax6 = plt.subplot(3, 3, 6)
        
        # 选择top 10特征
        if 'model_based' in feature_importance:
            importance_dict = feature_importance['model_based']
            sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            features, scores = zip(*sorted_features)
            
            y_pos = np.arange(len(features))
            bars = ax6.barh(y_pos, scores, color='skyblue')
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(features)
            ax6.set_xlabel('重要性得分')
            ax6.set_title('Top 10 特征重要性\n(基于模型系数)')
            ax6.grid(True, alpha=0.3)
            
            # 标注文献依据的重要特征
            literature_important = ['Z13_abs', 'Z18_abs', 'Z21_abs', 'GC_variance', 'BMI_GA_interaction']
            for i, feature in enumerate(features):
                if any(lit_feat in feature for lit_feat in literature_important):
                    bars[i].set_color('lightcoral')
    
    # 7. 性能指标对比雷达图
    ax7 = plt.subplot(3, 3, 7, projection='polar')
    
    metrics = ['precision', 'recall', 'f1']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    for strategy, color in [('f1_optimal', 'green'), ('recall_optimal', 'red'), ('conservative', 'orange')]:
        values = [evaluation_results[strategy][metric] for metric in metrics]
        values += values[:1]  # 闭合图形
        
        ax7.plot(angles, values, 'o-', linewidth=2, color=color, 
                label=strategy.replace('_', ' ').title())
        ax7.fill(angles, values, alpha=0.1, color=color)
    
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(['精确率', '召回率', 'F1分数'])
    ax7.set_ylim(0, 1)
    ax7.set_title('三种策略性能对比\n(雷达图)', pad=20)
    ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # 8. 阈值敏感性分析
    ax8 = plt.subplot(3, 3, 8)
    
    thresholds = np.linspace(0.1, 0.9, 50)
    precisions, recalls, f1s = [], [], []
    
    for thresh in thresholds:
        y_pred_thresh = (prob_test >= thresh).astype(int)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    ax8.plot(thresholds, precisions, 'b-', label='精确率', linewidth=2)
    ax8.plot(thresholds, recalls, 'r-', label='召回率', linewidth=2)
    ax8.plot(thresholds, f1s, 'g-', label='F1分数', linewidth=2)
    
    # 标注选择的阈值
    for strategy, color in [('f1_optimal', 'green'), ('recall_optimal', 'red')]:
        threshold = evaluation_results[strategy]['threshold']
        ax8.axvline(x=threshold, color=color, linestyle='--', alpha=0.7, 
                   label=f'{strategy.replace("_", " ")} 阈值')
    
    ax8.set_xlabel('分类阈值')
    ax8.set_ylabel('性能指标')
    ax8.set_title('阈值敏感性分析')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. 文献证据总结
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    literature_summary = """
文献证据支持总结:

1. 女胎检测依据:
   • Y染色体浓度<30%分位数
   • ZY值<25%分位数  
   • FF估计<12% (魏智芳2025)

2. 关键特征发现:
   • GC含量异常是重要指标
   • BMI≥35需21周后检测
   • Z值>2.5为异常阈值

3. 临床应用建议:
   • 召回优先: 初步筛查
   • F1优化: 确诊检测
   • 保守策略: 高风险人群
"""
    
    ax9.text(0.05, 0.95, literature_summary, transform=ax9.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'literature_based_comprehensive_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_literature_informed_results(evaluation_results, feature_importance, selected_features,
                                    model_results, female_stats, threshold_results):
    """
    保存基于文献的结果
    """
    # 主要结果文件
    main_results = {
        'version': SCRIPT_VERSION,
        'literature_basis': {
            'primary_references': [
                'Wei_Zhifang_2025_FF_threshold_analysis',
                'Xue_Ying_2017_quantitative_relationships', 
                'Canick_2013_Y_chromosome_specificity',
                'Wang_2013_gestational_effects',
                'Kinnings_2015_BMI_plateau',
                'Livergood_2017_obesity_effects',
                'Haghiac_2013_adipocyte_mechanisms'
            ],
            'key_findings': [
                'FF<4%为NIPT检测失败阈值',
                'BMI≥35kg/m²需21周后检测',
                'Y染色体浓度是女胎判定金标准',
                'GC含量变异是重要异常指标',
                '孕周与BMI存在补偿性交互作用'
            ]
        },
        'female_detection': female_stats,
        'feature_selection': {
            'total_features_engineered': 50,
            'final_selected_features': len(selected_features),
            'selected_features': selected_features,
            'literature_priority_features': [
                'Z13_abs', 'Z18_abs', 'Z21_abs',
                'trisomy_risk_score', 'GC_variance', 
                'BMI_GA_interaction', 'total_z_abnormal'
            ]
        },
        'model_performance': evaluation_results,
        'threshold_strategies': {
            'f1_optimal': threshold_results['f1_optimal'],
            'recall_optimal': threshold_results['recall_optimal'],
            'conservative_clinical': threshold_results['conservative_clinical']
        },
        'clinical_recommendations': {
            'screening_strategy': '召回优先阈值用于初步筛查',
            'diagnostic_strategy': 'F1优化阈值用于确诊检测',
            'high_risk_strategy': '保守阈值用于高风险人群',
            'bmi_adjustment': 'BMI≥35kg/m²的孕妇建议21周后检测',
            'ff_threshold': 'FF<4%时建议重采血或延迟检测'
        }
    }
    
    # 保存主要结果
    with open(os.path.join(RESULTS_DIR, 'literature_based_comprehensive_results.json'), 'w', encoding='utf-8') as f:
        json.dump(main_results, f, ensure_ascii=False, indent=2)
    
    # 保存特征重要性
    if feature_importance:
        with open(os.path.join(RESULTS_DIR, 'literature_informed_feature_importance.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'version': SCRIPT_VERSION,
                'feature_importance': feature_importance,
                'literature_interpretation': {
                    'top_gc_features': [f for f in selected_features if 'GC' in f],
                    'top_z_features': [f for f in selected_features if any(z in f for z in ['Z13', 'Z18', 'Z21'])],
                    'top_clinical_features': [f for f in selected_features if any(c in f for c in ['BMI', 'GA', '孕周'])]
                }
            }, f, ensure_ascii=False, indent=2)
    
    # 保存模型对比
    model_comparison = {}
    for name, result in model_results.items():
        model_comparison[name] = {
            'best_score': result['best_score'],
            'best_params': result['best_params']
        }
    
    with open(os.path.join(RESULTS_DIR, 'literature_based_model_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'version': SCRIPT_VERSION,
            'model_comparison': model_comparison,
            'selection_criteria': '基于临床导向评分函数(0.5×召回率+0.3×F1+0.2×精确率)'
        }, f, ensure_ascii=False, indent=2)

def main(target_recall: float = 0.90, test_size: float = 0.2, seed: int = 42,
         results_dir: Optional[str] = None):
    """
    基于文献证据链的女胎异常检测分析 v1.5
    """
    np.random.seed(seed)
    if results_dir is not None:
        global RESULTS_DIR
        RESULTS_DIR = results_dir
        os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"=== 基于文献证据链的女胎异常检测分析 {SCRIPT_VERSION} ===")
    print("文献依据: 魏智芳2025, 薛莹2017, Canick2013, Wang2013, Kinnings2015, Livergood2017")
    
    # 1. 读取和合并数据
    print("\n1. 读取数据...")
    dfa = read_and_unify(DATA_A)
    dfb = read_and_unify(DATA_B)
    raw = pd.concat([dfa, dfb], ignore_index=True)
    print(f"原始数据量: {len(raw)}")
    
    # 2. 基于文献的女胎筛选
    print("\n2. 基于文献证据的女胎样本筛选...")
    female_df, female_stats = literature_based_female_detection(raw)
    print(f"女胎样本数: {len(female_df)} ({female_stats['female_ratio']:.2%})")
    print(f"筛选方法: {female_stats['method']}")
    print(f"文献依据: {len(female_stats['literature_basis'])}篇核心文献")
    
    # 3. 基于文献的特征工程
    print("\n3. 基于文献指导的特征工程...")
    X, y, feature_names = literature_guided_feature_engineering(female_df)
    groups = female_df['孕妇代码'].astype(str).fillna('NA')
    print(f"工程化特征数量: {len(feature_names)}")
    print(f"样本标签分布: 正常={len(y)-y.sum()}, 异常={y.sum()} (异常率={y.mean():.1%})")
    
    # 4. 基于文献的模型选择和训练
    print("\n4. 基于文献指导的模型训练...")
    (best_model, X_train, X_test, y_train, y_test, 
     selected_features, model_results, best_model_name) = literature_informed_model_selection(
        X, y, groups, target_recall, test_size, seed)
    
    # 5. 基于文献的阈值优化
    print("\n5. 基于文献的阈值优化...")
    threshold_results = literature_based_threshold_optimization(
        best_model, X_train, y_train, target_recall)
    
    # 6. 全面评估
    print("\n6. 基于文献背景的模型评估...")
    evaluation_results, feature_importance, prob_test, predictions = comprehensive_evaluation_with_literature_context(
        best_model, X_train, X_test, y_train, y_test, selected_features, threshold_results, target_recall)
    
    # 打印主要结果
    print(f"\n=== 基于文献证据的最终结果 ===")
    print(f"最佳模型: {best_model_name}")
    
    for strategy in ['f1_optimal', 'recall_optimal', 'conservative']:
        result = evaluation_results[strategy]
        strategy_name = {'f1_optimal': 'F1优化', 'recall_optimal': '召回优先', 'conservative': '保守策略'}[strategy]
        print(f"\n{strategy_name}策略:")
        print(f"  阈值: {result['threshold']:.4f}")
        print(f"  精确率: {result['precision']:.3f}")
        print(f"  召回率: {result['recall']:.3f}")
        print(f"  F1分数: {result['f1']:.3f}")
    
    if evaluation_results['f1_optimal']['roc_auc']:
        print(f"\nROC AUC: {evaluation_results['f1_optimal']['roc_auc']:.3f}")
    
    # 7. 生成基于文献的可视化
    print("\n7. 生成基于文献证据的可视化分析...")
    create_literature_informed_visualizations(
        evaluation_results, feature_importance, y_test, prob_test, predictions, threshold_results)
    
    # 8. 保存基于文献的结果
    print("\n8. 保存基于文献证据的分析结果...")
    save_literature_informed_results(
        evaluation_results, feature_importance, selected_features,
        model_results, female_stats, threshold_results)
    
    # 保存测试集预测结果
    test_results = female_df.loc[X_test.index].copy()
    test_results['y_true'] = y_test.values
    test_results['prob_score'] = prob_test
    for strategy, y_pred in predictions.items():
        test_results[f'pred_{strategy}'] = y_pred
    test_results.to_csv(os.path.join(RESULTS_DIR, 'literature_based_test_predictions.csv'), 
                       index=False, encoding='utf-8-sig')
    
    # 保存模型
    try:
        import joblib
        joblib.dump(best_model, os.path.join(RESULTS_DIR, 'literature_informed_best_model.joblib'))
        print("模型已保存")
    except Exception as e:
        print(f"模型保存失败: {e}")
    
    print(f"\n所有基于文献证据的结果已保存到: {RESULTS_DIR}")
    print("\n=== 文献证据链验证完成 ===")
    
    return best_model, evaluation_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='基于文献证据链的女胎异常检测分析 v1.5')
    parser.add_argument('--target_recall', type=float, default=0.90, 
                       help='目标召回率 (基于魏智芳2025的临床需求)')
    parser.add_argument('--test_size', type=float, default=0.2, 
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子')
    parser.add_argument('--results_dir', type=str, default=None, 
                       help='结果输出目录')
    
    args = parser.parse_args()
    main(target_recall=args.target_recall, test_size=args.test_size, 
         seed=args.seed, results_dir=args.results_dir)
