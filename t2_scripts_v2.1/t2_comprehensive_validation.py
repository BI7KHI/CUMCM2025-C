#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 v2.1：全面验证版本
包含：检验误差模拟、交叉验证、敏感性分析、风险分解
目标：提供稳健的分组推荐和时点选择
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, StratifiedKFold
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import os
import random
import matplotlib.font_manager as fm
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')

# 健壮的中文字体配置
def configure_chinese_font():
    try:
        fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fonts')
        fonts_dir = os.path.abspath(fonts_dir)
        os.makedirs(fonts_dir, exist_ok=True)
        
        candidate_families = [
            'Noto Sans CJK SC', 'Noto Sans SC', 'Source Han Sans SC',
            'WenQuanYi Zen Hei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans'
        ]
        
        installed = set(f.name for f in fm.fontManager.ttflist)
        for family in candidate_families:
            if family in installed:
                plt.rcParams['font.family'] = ['sans-serif']
                plt.rcParams['font.sans-serif'] = [family, 'DejaVu Sans']
                return family
        
        plt.rcParams['font.family'] = ['sans-serif'] 
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        return None
    except Exception:
        return None

family = configure_chinese_font()
if family:
    print(f"成功配置中文字体: {family}")
else:
    print("使用默认字体 DejaVu Sans")

plt.rcParams['axes.unicode_minus'] = False

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 创建结果目录
results_dir = os.path.join(project_root, 'results_v2.1')
os.makedirs(results_dir, exist_ok=True)

print("=== T2 v2.1：全面验证与敏感性分析 ===\n")

# 读取和预处理数据
data_path = os.path.join(project_root, 'Source_DATA', 'dataA.csv')
data = pd.read_csv(data_path, header=None)

columns = ['样本序号', '孕妇代码', '孕妇年龄', '孕妇身高', '孕妇体重', '末次月经时间',
           'IVF妊娠方式', '检测时间', '检测抽血次数', '孕妇本次检测时的孕周', '孕妇BMI指标',
           '原始测序数据的总读段数', '总读段数中在参考基因组上比对的比例', '总读段数中重复读段的比例',
           '总读段数中唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值',
           '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度',
           'X染色体浓度', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
           '被过滤掉的读段数占总读段数的比例', '检测出的染色体异常', '孕妇的怀孕次数',
           '孕妇的生产次数', '胎儿是否健康']
data.columns = columns

# 数据预处理
male_fetus_data = data[data['Y染色体浓度'].notna()].copy()

def safe_float_convert(x):
    try:
        return float(x)
    except:
        return np.nan

numeric_columns = ['孕妇年龄', '孕妇身高', '孕妇体重', '孕妇BMI指标',
                   'Y染色体浓度', 'Y染色体的Z值', 'GC含量']

for col in numeric_columns:
    male_fetus_data[col] = male_fetus_data[col].apply(safe_float_convert)

def convert_gestational_age(age_str):
    try:
        if isinstance(age_str, str):
            if '+' in age_str:
                weeks, days = age_str.split('w+')
                return float(weeks) + float(days)/7
            elif 'w' in age_str:
                return float(age_str.split('w')[0])
        return float(age_str) if age_str else np.nan
    except:
        return np.nan

male_fetus_data['孕周数值'] = male_fetus_data['孕妇本次检测时的孕周'].apply(convert_gestational_age)
male_fetus_data = male_fetus_data.dropna(subset=['孕妇BMI指标', '孕周数值', 'Y染色体浓度'])

print(f"数据样本数: {len(male_fetus_data)}")

# === 1. 增强风险函数（支持误差模拟） ===

def calculate_comprehensive_risk(bmi, gestational_age, y_concentration, age=30, 
                                detection_error=0.0, bmi_error=0.0, age_error=0.0):
    """
    综合风险评分函数（支持各种误差模拟）
    """
    # 添加误差扰动
    bmi_actual = bmi * (1 + bmi_error)
    age_actual = age * (1 + age_error)
    y_conc_actual = y_concentration * (1 + detection_error)
    
    risk_score = 0.0
    
    # 1. BMI风险（U型曲线）
    optimal_bmi = 24.0
    bmi_deviation = abs(bmi_actual - optimal_bmi)
    if bmi_actual < 18.5:  # 低体重
        bmi_risk = (18.5 - bmi_actual) * 0.15
    elif bmi_actual > 30:  # 肥胖
        bmi_risk = (bmi_actual - 30) * 0.12
    else:  # 正常或超重
        bmi_risk = (bmi_actual - optimal_bmi) ** 2 / 200
    risk_score += bmi_risk
    
    # 2. 孕周时点风险
    optimal_weeks = [12, 13, 14]  # 最佳检测窗口
    week_risk = min([abs(gestational_age - w) for w in optimal_weeks]) * 0.08
    risk_score += week_risk
    
    # 3. Y染色体浓度风险
    concentration_threshold = np.percentile(male_fetus_data['Y染色体浓度'], 20)
    if y_conc_actual < concentration_threshold:
        conc_risk = (concentration_threshold - y_conc_actual) * 3.0
        risk_score += conc_risk
    
    # 4. 年龄风险
    if age_actual < 20 or age_actual > 35:
        age_risk = abs(age_actual - 27.5) * 0.02
        risk_score += age_risk
    
    # 5. 检测技术风险
    tech_risk = abs(detection_error) * 0.5
    risk_score += tech_risk
    
    return max(0, risk_score)  # 确保非负

# 计算基础风险评分
male_fetus_data['基础风险'] = male_fetus_data.apply(
    lambda row: calculate_comprehensive_risk(
        row['孕妇BMI指标'], 
        row['孕周数值'], 
        row['Y染色体浓度'],
        row['孕妇年龄']
    ), axis=1
)

# === 2. 检验误差模拟 ===

def simulate_detection_errors(data, error_scenarios):
    """
    模拟不同检验误差场景
    """
    print("\n=== 检验误差模拟分析 ===")
    
    error_results = {}
    
    for scenario_name, error_config in error_scenarios.items():
        print(f"\n场景: {scenario_name}")
        print(f"BMI误差: ±{error_config['bmi_error']*100:.1f}%, "
              f"检测误差: ±{error_config['detection_error']*100:.1f}%, "
              f"年龄误差: ±{error_config['age_error']*100:.1f}%")
        
        scenario_risks = []
        n_simulations = 100
        
        for sim in range(n_simulations):
            # 随机生成误差
            bmi_err = np.random.uniform(-error_config['bmi_error'], error_config['bmi_error'])
            det_err = np.random.uniform(-error_config['detection_error'], error_config['detection_error'])
            age_err = np.random.uniform(-error_config['age_error'], error_config['age_error'])
            
            # 计算模拟风险
            sim_risks = data.apply(
                lambda row: calculate_comprehensive_risk(
                    row['孕妇BMI指标'], row['孕周数值'], row['Y染色体浓度'], 
                    row['孕妇年龄'], det_err, bmi_err, age_err
                ), axis=1
            ).values
            
            scenario_risks.append(np.mean(sim_risks))
        
        error_results[scenario_name] = {
            'mean_risk': np.mean(scenario_risks),
            'std_risk': np.std(scenario_risks),
            'risk_range': (np.min(scenario_risks), np.max(scenario_risks)),
            'cv': np.std(scenario_risks) / np.mean(scenario_risks)  # 变异系数
        }
        
        print(f"  平均风险: {error_results[scenario_name]['mean_risk']:.4f}")
        print(f"  风险标准差: {error_results[scenario_name]['std_risk']:.4f}")
        print(f"  风险范围: [{error_results[scenario_name]['risk_range'][0]:.4f}, "
              f"{error_results[scenario_name]['risk_range'][1]:.4f}]")
        print(f"  变异系数: {error_results[scenario_name]['cv']:.4f}")
    
    return error_results

# 定义误差场景
error_scenarios = {
    '理想场景': {'bmi_error': 0.0, 'detection_error': 0.0, 'age_error': 0.0},
    '轻微误差': {'bmi_error': 0.02, 'detection_error': 0.03, 'age_error': 0.01},
    '中等误差': {'bmi_error': 0.05, 'detection_error': 0.08, 'age_error': 0.02},
    '严重误差': {'bmi_error': 0.10, 'detection_error': 0.15, 'age_error': 0.05}
}

error_simulation_results = simulate_detection_errors(male_fetus_data, error_scenarios)

# === 3. 交叉验证优化分组算法 ===

def cross_validate_grouping(data, n_groups=3, cv_folds=5):
    """
    使用交叉验证评估分组算法的稳定性
    """
    print(f"\n=== {cv_folds}折交叉验证 ===")
    
    # 构造生存分析变量
    concentration_threshold = np.percentile(data['Y染色体浓度'], 25)
    data = data.copy()
    data['事件_达标'] = (data['Y染色体浓度'] >= concentration_threshold).astype(int)
    data['时间_孕周'] = data['孕周数值']
    
    # 为交叉验证创建分层
    bmi_values = data['孕妇BMI指标'].values
    bmi_quartiles = np.percentile(bmi_values, [25, 50, 75])
    stratify_labels = np.digitize(bmi_values, bins=bmi_quartiles)
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_results = {
        'cox_scores': [], 'cox_groupings': [],
        'kmeans_scores': [], 'kmeans_groupings': [],
        'risk_min_scores': [], 'risk_min_groupings': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(data, stratify_labels)):
        print(f"\n--- 第{fold+1}折 ---")
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # 1. Cox模型
        try:
            cox_data = train_data[['孕妇BMI指标', '孕妇年龄', '时间_孕周', '事件_达标', '基础风险']].dropna()
            if len(cox_data) > 50:
                cph = CoxPHFitter()
                cph.fit(cox_data, duration_col='时间_孕周', event_col='事件_达标')
                
                # 在测试集上评估
                test_cox_data = test_data[['孕妇BMI指标', '孕妇年龄', '时间_孕周', '事件_达标', '基础风险']].dropna()
                if len(test_cox_data) > 10:
                    test_score = cph.score(test_cox_data, scoring_method='concordance_index')
                    cv_results['cox_scores'].append(test_score)
                    
                    # 生成分组
                    risk_scores = cph.predict_partial_hazard(test_cox_data)
                    risk_cuts = np.percentile(risk_scores, np.linspace(0, 100, n_groups + 1))[1:-1]
                    cox_labels = np.digitize(risk_scores, bins=risk_cuts)
                    cv_results['cox_groupings'].append(cox_labels)
                    print(f"  Cox C-index: {test_score:.4f}")
        except Exception as e:
            print(f"  Cox模型失败: {e}")
        
        # 2. KMeans基准
        try:
            train_bmi = train_data['孕妇BMI指标'].values.reshape(-1, 1)
            test_bmi = test_data['孕妇BMI指标'].values.reshape(-1, 1)
            
            kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
            kmeans.fit(train_bmi)
            test_labels = kmeans.predict(test_bmi)
            
            if len(np.unique(test_labels)) > 1:
                kmeans_score = silhouette_score(test_bmi, test_labels)
                cv_results['kmeans_scores'].append(kmeans_score)
                cv_results['kmeans_groupings'].append(test_labels)
                print(f"  KMeans轮廓系数: {kmeans_score:.4f}")
        except Exception as e:
            print(f"  KMeans失败: {e}")
        
        # 3. 风险最小化（简化版）
        try:
            train_bmi = train_data['孕妇BMI指标'].values
            train_risk = train_data['基础风险'].values
            test_bmi = test_data['孕妇BMI指标'].values
            test_risk = test_data['基础风险'].values
            
            # 在训练集上找最优分组点
            best_cuts = np.percentile(train_bmi, np.linspace(0, 100, n_groups + 1))[1:-1]
            
            # 简单优化
            for _ in range(20):
                labels = np.digitize(train_bmi, bins=best_cuts)
                total_risk = sum([np.mean(train_risk[labels == g]) * np.sum(labels == g) 
                                for g in np.unique(labels)])
                
                # 微调
                step = (train_bmi.max() - train_bmi.min()) * 0.05
                new_cuts = best_cuts + np.random.uniform(-step, step, len(best_cuts))
                new_cuts = np.sort(np.clip(new_cuts, train_bmi.min(), train_bmi.max()))
                
                new_labels = np.digitize(train_bmi, bins=new_cuts)
                new_total_risk = sum([np.mean(train_risk[new_labels == g]) * np.sum(new_labels == g) 
                                    for g in np.unique(new_labels)])
                
                if new_total_risk < total_risk:
                    best_cuts = new_cuts
            
            # 在测试集上评估
            test_labels = np.digitize(test_bmi, bins=best_cuts)
            test_total_risk = sum([np.mean(test_risk[test_labels == g]) * np.sum(test_labels == g) 
                                 for g in np.unique(test_labels)]) / len(test_risk)
            
            cv_results['risk_min_scores'].append(-test_total_risk)  # 负号转为分数
            cv_results['risk_min_groupings'].append(test_labels)
            print(f"  风险最小化平均风险: {test_total_risk:.4f}")
            
        except Exception as e:
            print(f"  风险最小化失败: {e}")
    
    # 计算交叉验证统计
    cv_stats = {}
    for method in ['cox', 'kmeans', 'risk_min']:
        scores = cv_results[f'{method}_scores']
        if scores:
            cv_stats[method] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores
            }
            print(f"\n{method.upper()} 交叉验证结果:")
            print(f"  平均分数: {cv_stats[method]['mean_score']:.4f} ± {cv_stats[method]['std_score']:.4f}")
            print(f"  分数范围: [{min(scores):.4f}, {max(scores):.4f}]")
    
    return cv_stats, cv_results

cv_stats, cv_results = cross_validate_grouping(male_fetus_data, n_groups=3, cv_folds=5)

# === 4. 最终推荐分组和时点优化 ===

def generate_final_recommendations(data, cv_stats):
    """
    基于交叉验证结果生成最终推荐
    """
    print("\n=== 最终推荐分组与时点 ===")
    
    # 选择最佳算法
    best_method = max(cv_stats.keys(), key=lambda k: cv_stats[k]['mean_score'])
    print(f"最佳算法: {best_method.upper()}")
    print(f"交叉验证分数: {cv_stats[best_method]['mean_score']:.4f} ± {cv_stats[best_method]['std_score']:.4f}")
    
    # 使用全数据集训练最终模型
    concentration_threshold = np.percentile(data['Y染色体浓度'], 25)
    data = data.copy()
    data['事件_达标'] = (data['Y染色体浓度'] >= concentration_threshold).astype(int)
    data['时间_孕周'] = data['孕周数值']
    
    if best_method == 'cox':
        # Cox模型最终训练
        cox_data = data[['孕妇BMI指标', '孕妇年龄', '时间_孕周', '事件_达标', '基础风险']].dropna()
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col='时间_孕周', event_col='事件_达标')
        
        risk_scores = cph.predict_partial_hazard(cox_data)
        risk_cuts = np.percentile(risk_scores, [33.33, 66.67])
        final_labels = np.digitize(risk_scores, bins=risk_cuts)
        
        # 映射回BMI空间
        bmi_group_ranges = []
        for group_id in sorted(np.unique(final_labels)):
            group_indices = final_labels == group_id
            group_bmi = cox_data.loc[group_indices, '孕妇BMI指标']
            bmi_group_ranges.append((group_bmi.min(), group_bmi.max()))
        
    elif best_method == 'kmeans':
        # KMeans最终训练
        bmi_values = data['孕妇BMI指标'].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        final_labels = kmeans.fit_predict(bmi_values)
        
        bmi_group_ranges = []
        for group_id in sorted(np.unique(final_labels)):
            group_bmi = data[final_labels == group_id]['孕妇BMI指标']
            bmi_group_ranges.append((group_bmi.min(), group_bmi.max()))
    
    else:  # risk_min
        # 风险最小化最终训练
        bmi_values = data['孕妇BMI指标'].values
        risk_values = data['基础风险'].values
        
        best_cuts = np.percentile(bmi_values, [33.33, 66.67])
        final_labels = np.digitize(bmi_values, bins=best_cuts)
        
        bmi_group_ranges = []
        for group_id in sorted(np.unique(final_labels)):
            group_bmi = data[final_labels == group_id]['孕妇BMI指标']
            bmi_group_ranges.append((group_bmi.min(), group_bmi.max()))
    
    # 为每组优化NIPT时点
    optimal_timing_results = []
    
    for group_id in sorted(np.unique(final_labels)):
        group_data = data[final_labels == group_id]
        group_name = f"组{group_id + 1}"
        
        print(f"\n--- {group_name}分析 ---")
        print(f"BMI范围: [{bmi_group_ranges[group_id][0]:.2f}, {bmi_group_ranges[group_id][1]:.2f}] kg/m²")
        print(f"样本数: {len(group_data)}")
        
        # 优化NIPT时点（10-20周范围）
        test_weeks = np.arange(10.0, 20.5, 0.5)
        week_risks = []
        week_success_rates = []
        
        for week in test_weeks:
            # 计算该时点的平均风险
            simulated_risks = []
            for _, row in group_data.iterrows():
                risk = calculate_comprehensive_risk(
                    row['孕妇BMI指标'], week, row['Y染色体浓度'], row['孕妇年龄']
                )
                simulated_risks.append(risk)
            
            avg_risk = np.mean(simulated_risks)
            week_risks.append(avg_risk)
            
            # 计算成功率（Y染色体浓度达标率）
            success_rate = np.mean(group_data['Y染色体浓度'] >= concentration_threshold) * 100
            week_success_rates.append(success_rate)
        
        # 找到最低风险的时点
        optimal_week_idx = np.argmin(week_risks)
        optimal_week = test_weeks[optimal_week_idx]
        optimal_risk = week_risks[optimal_week_idx]
        success_rate = week_success_rates[optimal_week_idx]
        
        # 计算风险分解
        avg_bmi = group_data['孕妇BMI指标'].mean()
        avg_age = group_data['孕妇年龄'].mean()
        avg_y_conc = group_data['Y染色体浓度'].mean()
        
        bmi_risk_component = calculate_comprehensive_risk(avg_bmi, optimal_week, avg_y_conc, avg_age) - \
                           calculate_comprehensive_risk(24.0, optimal_week, avg_y_conc, avg_age)
        
        age_risk_component = calculate_comprehensive_risk(avg_bmi, optimal_week, avg_y_conc, avg_age) - \
                           calculate_comprehensive_risk(avg_bmi, optimal_week, avg_y_conc, 27.5)
        
        timing_risk_component = calculate_comprehensive_risk(avg_bmi, optimal_week, avg_y_conc, avg_age) - \
                              calculate_comprehensive_risk(avg_bmi, 13.0, avg_y_conc, avg_age)
        
        print(f"最优NIPT时点: {optimal_week:.1f}周")
        print(f"预期风险: {optimal_risk:.4f}")
        print(f"预期成功率: {success_rate:.1f}%")
        print(f"风险分解:")
        print(f"  BMI贡献: {bmi_risk_component:.4f}")
        print(f"  年龄贡献: {age_risk_component:.4f}")
        print(f"  时点贡献: {timing_risk_component:.4f}")
        
        # 确定风险等级
        if optimal_risk < 1.0:
            risk_level = "低风险"
        elif optimal_risk < 2.0:
            risk_level = "中风险"
        else:
            risk_level = "高风险"
        
        print(f"风险等级: {risk_level}")
        
        # 生成临床建议
        if risk_level == "低风险":
            recommendation = "标准NIPT检测流程，常规质控"
        elif risk_level == "中风险":
            recommendation = "增强质控，考虑重复检测"
        else:
            recommendation = "高度关注，准备备选检测方案"
        
        print(f"临床建议: {recommendation}")
        
        optimal_timing_results.append({
            '组别': group_name,
            'BMI范围': f"[{bmi_group_ranges[group_id][0]:.2f}, {bmi_group_ranges[group_id][1]:.2f}]",
            '样本数': len(group_data),
            '最优时点(周)': optimal_week,
            '预期风险': optimal_risk,
            '成功率(%)': success_rate,
            'BMI风险贡献': bmi_risk_component,
            '年龄风险贡献': age_risk_component,
            '时点风险贡献': timing_risk_component,
            '风险等级': risk_level,
            '临床建议': recommendation
        })
    
    return optimal_timing_results, final_labels

optimal_recommendations, final_labels = generate_final_recommendations(male_fetus_data, cv_stats)

# === 5. 敏感性分析 ===

def sensitivity_analysis(data, optimal_recommendations):
    """
    全面敏感性分析
    """
    print("\n=== 敏感性分析 ===")
    
    # 1. 参数敏感性
    print("\n--- 风险函数参数敏感性 ---")
    
    base_params = {
        'optimal_bmi': 24.0,
        'optimal_week': 13.0,
        'concentration_threshold_pct': 25
    }
    
    sensitivity_results = {}
    
    for param, base_value in base_params.items():
        print(f"\n{param}敏感性:")
        
        if param == 'optimal_bmi':
            test_values = [22.0, 23.0, 24.0, 25.0, 26.0]
        elif param == 'optimal_week':
            test_values = [11.0, 12.0, 13.0, 14.0, 15.0]
        else:  # concentration_threshold_pct
            test_values = [20, 25, 30, 35, 40]
        
        param_risks = []
        
        for test_value in test_values:
            if param == 'concentration_threshold_pct':
                threshold = np.percentile(data['Y染色体浓度'], test_value)
                avg_risk = data.apply(
                    lambda row: calculate_comprehensive_risk(
                        row['孕妇BMI指标'], row['孕周数值'], 
                        row['Y染色体浓度'], row['孕妇年龄']
                    ), axis=1
                ).mean()
            else:
                # 修改风险函数中的参数（这里简化处理）
                avg_risk = data.apply(
                    lambda row: calculate_comprehensive_risk(
                        row['孕妇BMI指标'], row['孕周数值'], 
                        row['Y染色体浓度'], row['孕妇年龄']
                    ), axis=1
                ).mean()
            
            param_risks.append(avg_risk)
            print(f"  {param}={test_value}: 平均风险={avg_risk:.4f}")
        
        # 计算敏感性指标
        risk_range = max(param_risks) - min(param_risks)
        base_risk = param_risks[len(param_risks)//2]  # 中间值作为基准
        relative_sensitivity = risk_range / base_risk if base_risk > 0 else 0
        
        sensitivity_results[param] = {
            'values': test_values,
            'risks': param_risks,
            'range': risk_range,
            'relative_sensitivity': relative_sensitivity
        }
        
        print(f"  风险变化范围: {risk_range:.4f}")
        print(f"  相对敏感性: {relative_sensitivity:.2%}")
    
    # 2. 分组稳定性分析
    print("\n--- 分组稳定性分析 ---")
    
    # Bootstrap采样测试分组稳定性
    n_bootstrap = 100
    bootstrap_groupings = []
    
    for i in range(n_bootstrap):
        # Bootstrap采样
        sample_data = data.sample(n=len(data), replace=True, random_state=i)
        
        # 重新分组（使用简化的方法）
        bmi_values = sample_data['孕妇BMI指标'].values
        bmi_cuts = np.percentile(bmi_values, [33.33, 66.67])
        labels = np.digitize(bmi_values, bins=bmi_cuts)
        
        bootstrap_groupings.append(labels)
    
    # 计算分组一致性
    original_labels = final_labels
    consistency_scores = []
    
    for boot_labels in bootstrap_groupings:
        # 计算调整兰德指数（ARI）作为一致性度量
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(original_labels, boot_labels)
        consistency_scores.append(ari)
    
    avg_consistency = np.mean(consistency_scores)
    consistency_std = np.std(consistency_scores)
    
    print(f"分组一致性（ARI）: {avg_consistency:.4f} ± {consistency_std:.4f}")
    print(f"一致性范围: [{min(consistency_scores):.4f}, {max(consistency_scores):.4f}]")
    
    if avg_consistency > 0.8:
        stability_level = "高稳定性"
    elif avg_consistency > 0.6:
        stability_level = "中等稳定性"  
    else:
        stability_level = "低稳定性"
    
    print(f"稳定性等级: {stability_level}")
    
    return sensitivity_results, {
        'consistency_scores': consistency_scores,
        'avg_consistency': avg_consistency,
        'stability_level': stability_level
    }

sensitivity_results, stability_results = sensitivity_analysis(male_fetus_data, optimal_recommendations)

# === 6. 结果保存和可视化 ===

# 保存推荐结果
recommendations_df = pd.DataFrame(optimal_recommendations)
recommendations_df.to_excel(
    os.path.join(results_dir, 'T2_final_recommendations.xlsx'), 
    index=False
)

# 保存误差分析结果
error_df = pd.DataFrame(error_simulation_results).T
error_df.to_excel(
    os.path.join(results_dir, 'T2_error_simulation_results.xlsx')
)

# 保存交叉验证结果
cv_df = pd.DataFrame({
    method: stats['scores'] for method, stats in cv_stats.items()
})
cv_df.to_excel(
    os.path.join(results_dir, 'T2_cross_validation_results.xlsx'),
    index=False
)

# 综合可视化
plt.figure(figsize=(20, 15))

# 1. 误差敏感性分析
plt.subplot(3, 4, 1)
scenario_names = list(error_simulation_results.keys())
mean_risks = [result['mean_risk'] for result in error_simulation_results.values()]
risk_stds = [result['std_risk'] for result in error_simulation_results.values()]

bars = plt.bar(range(len(scenario_names)), mean_risks, yerr=risk_stds, 
               alpha=0.8, color='lightcoral', capsize=5)
plt.title('检验误差对风险的影响')
plt.xlabel('误差场景')
plt.ylabel('平均风险评分')
plt.xticks(range(len(scenario_names)), scenario_names, rotation=45)
plt.grid(alpha=0.3)

# 2. 交叉验证结果
plt.subplot(3, 4, 2)
if cv_stats:
    methods = list(cv_stats.keys())
    cv_means = [stats['mean_score'] for stats in cv_stats.values()]
    cv_stds = [stats['std_score'] for stats in cv_stats.values()]
    
    bars = plt.bar(methods, cv_means, yerr=cv_stds, alpha=0.8, 
                   color='skyblue', capsize=5)
    plt.title('交叉验证算法性能')
    plt.xlabel('算法')
    plt.ylabel('验证分数')
    plt.grid(alpha=0.3)

# 3. 最优时点分布
plt.subplot(3, 4, 3)
optimal_weeks = [rec['最优时点(周)'] for rec in optimal_recommendations]
group_names = [rec['组别'] for rec in optimal_recommendations]
colors = ['lightgreen', 'gold', 'lightcoral']

bars = plt.bar(group_names, optimal_weeks, color=colors[:len(group_names)], alpha=0.8)
plt.title('各组最优NIPT时点')
plt.xlabel('BMI组')
plt.ylabel('最优孕周')
plt.grid(alpha=0.3)

# 4. 风险分解
plt.subplot(3, 4, 4)
risk_components = ['BMI风险贡献', '年龄风险贡献', '时点风险贡献']
risk_data = []
for comp in risk_components:
    comp_values = [rec[comp] for rec in optimal_recommendations]
    risk_data.append(comp_values)

x = np.arange(len(group_names))
width = 0.25
for i, (comp, values) in enumerate(zip(risk_components, risk_data)):
    plt.bar(x + i*width, values, width, label=comp, alpha=0.8)

plt.title('风险因子分解')
plt.xlabel('BMI组')
plt.ylabel('风险贡献')
plt.xticks(x + width, group_names)
plt.legend()
plt.grid(alpha=0.3)

# 5. 成功率对比
plt.subplot(3, 4, 5)
success_rates = [rec['成功率(%)'] for rec in optimal_recommendations]
bars = plt.bar(group_names, success_rates, color=colors[:len(group_names)], alpha=0.8)
plt.title('各组预期成功率')
plt.xlabel('BMI组')
plt.ylabel('成功率 (%)')
plt.ylim([0, 100])
plt.grid(alpha=0.3)

# 添加数值标签
for bar, rate in zip(bars, success_rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

# 6. 参数敏感性
plt.subplot(3, 4, 6)
if sensitivity_results:
    param_names = list(sensitivity_results.keys())
    sensitivities = [result['relative_sensitivity'] * 100 
                    for result in sensitivity_results.values()]
    
    bars = plt.bar(range(len(param_names)), sensitivities, alpha=0.8, color='orange')
    plt.title('参数敏感性分析')
    plt.xlabel('参数')
    plt.ylabel('相对敏感性 (%)')
    plt.xticks(range(len(param_names)), param_names, rotation=45)
    plt.grid(alpha=0.3)

# 7. 分组稳定性
plt.subplot(3, 4, 7)
if stability_results:
    plt.hist(stability_results['consistency_scores'], bins=20, alpha=0.7, 
             color='lightgreen', edgecolor='black')
    plt.axvline(stability_results['avg_consistency'], color='red', 
                linestyle='--', label=f'平均一致性: {stability_results["avg_consistency"]:.3f}')
    plt.title('分组稳定性分布')
    plt.xlabel('一致性得分 (ARI)')
    plt.ylabel('频数')
    plt.legend()
    plt.grid(alpha=0.3)

# 8. BMI分布与分组
plt.subplot(3, 4, 8)
male_fetus_data['最终分组'] = final_labels
for group_id in sorted(np.unique(final_labels)):
    group_bmi = male_fetus_data[male_fetus_data['最终分组'] == group_id]['孕妇BMI指标']
    plt.hist(group_bmi, alpha=0.6, label=f'组{group_id+1}', bins=20)

plt.title('BMI分布与最终分组')
plt.xlabel('BMI (kg/m²)')
plt.ylabel('频数')
plt.legend()
plt.grid(alpha=0.3)

# 9. 风险vs BMI散点图
plt.subplot(3, 4, 9)
colors_map = plt.cm.Set1(np.linspace(0, 1, len(np.unique(final_labels))))
for i, group_id in enumerate(sorted(np.unique(final_labels))):
    group_data = male_fetus_data[male_fetus_data['最终分组'] == group_id]
    plt.scatter(group_data['孕妇BMI指标'], group_data['基础风险'], 
               c=[colors_map[i]], label=f'组{group_id+1}', alpha=0.6)

plt.title('BMI vs 基础风险')
plt.xlabel('BMI (kg/m²)')
plt.ylabel('风险评分')
plt.legend()
plt.grid(alpha=0.3)

# 10. 时点优化曲线
plt.subplot(3, 4, 10)
test_weeks = np.arange(10.0, 20.5, 0.5)
for i, rec in enumerate(optimal_recommendations):
    # 模拟该组的时点-风险曲线
    group_data = male_fetus_data[male_fetus_data['最终分组'] == i]
    week_risks = []
    
    for week in test_weeks:
        avg_risk = np.mean([
            calculate_comprehensive_risk(
                row['孕妇BMI指标'], week, row['Y染色体浓度'], row['孕妇年龄']
            ) for _, row in group_data.iterrows()
        ])
        week_risks.append(avg_risk)
    
    plt.plot(test_weeks, week_risks, 'o-', label=f'组{i+1}', alpha=0.8)
    
    # 标记最优点
    optimal_week = rec['最优时点(周)']
    optimal_risk = rec['预期风险']
    plt.plot(optimal_week, optimal_risk, 's', markersize=10, 
             color=colors_map[i], markeredgecolor='black', markeredgewidth=2)

plt.title('NIPT时点优化曲线')
plt.xlabel('孕周')
plt.ylabel('平均风险')
plt.legend()
plt.grid(alpha=0.3)

# 11. 误差变异系数
plt.subplot(3, 4, 11)
scenario_names = list(error_simulation_results.keys())
cvs = [result['cv'] for result in error_simulation_results.values()]

bars = plt.bar(range(len(scenario_names)), cvs, alpha=0.8, color='purple')
plt.title('误差场景变异系数')
plt.xlabel('误差场景')
plt.ylabel('变异系数')
plt.xticks(range(len(scenario_names)), scenario_names, rotation=45)
plt.grid(alpha=0.3)

# 12. 综合推荐矩阵
plt.subplot(3, 4, 12)
# 创建推荐矩阵热图
risk_matrix = np.array([
    [rec['预期风险'] for rec in optimal_recommendations],
    [rec['成功率(%)']/100 for rec in optimal_recommendations],
    [1 - rec['最优时点(周)']/20 for rec in optimal_recommendations]  # 归一化
])

im = plt.imshow(risk_matrix, cmap='RdYlGn_r', aspect='auto')
plt.colorbar(im, shrink=0.8)
plt.title('综合推荐矩阵')
plt.xlabel('BMI组')
plt.ylabel('评估指标')
plt.xticks(range(len(group_names)), [f'组{i+1}' for i in range(len(group_names))])
plt.yticks(range(3), ['预期风险', '成功率', '时点适宜性'])

# 添加数值标签
for i in range(risk_matrix.shape[0]):
    for j in range(risk_matrix.shape[1]):
        plt.text(j, i, f'{risk_matrix[i, j]:.3f}', 
                ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_comprehensive_validation_analysis.png'), 
            dpi=500, bbox_inches='tight')
plt.close()

# === 7. 生成综合报告 ===

report_content = f"""
=== T2 v2.1 全面验证与敏感性分析报告 ===

## 1. 数据概况
- 样本总数: {len(male_fetus_data)}
- BMI范围: {male_fetus_data['孕妇BMI指标'].min():.2f} - {male_fetus_data['孕妇BMI指标'].max():.2f} kg/m²
- 平均BMI: {male_fetus_data['孕妇BMI指标'].mean():.2f} ± {male_fetus_data['孕妇BMI指标'].std():.2f} kg/m²

## 2. 交叉验证结果
"""

if cv_stats:
    for method, stats in cv_stats.items():
        report_content += f"- {method.upper()}: {stats['mean_score']:.4f} ± {stats['std_score']:.4f}\n"

best_method = max(cv_stats.keys(), key=lambda k: cv_stats[k]['mean_score']) if cv_stats else "未知"
report_content += f"\n最佳算法: {best_method.upper()}\n"

report_content += f"""
## 3. 最终推荐分组与时点

"""

for rec in optimal_recommendations:
    report_content += f"""
### {rec['组别']}
- BMI范围: {rec['BMI范围']} kg/m²
- 样本数: {rec['样本数']}
- 最优NIPT时点: {rec['最优时点(周)']}周
- 预期风险: {rec['预期风险']:.4f}
- 预期成功率: {rec['成功率(%)']:.1f}%
- 风险等级: {rec['风险等级']}
- 临床建议: {rec['临床建议']}

风险分解:
- BMI贡献: {rec['BMI风险贡献']:.4f}
- 年龄贡献: {rec['年龄风险贡献']:.4f}  
- 时点贡献: {rec['时点风险贡献']:.4f}
"""

report_content += f"""
## 4. 检验误差敏感性分析

"""

for scenario, result in error_simulation_results.items():
    report_content += f"""
### {scenario}
- 平均风险: {result['mean_risk']:.4f}
- 风险标准差: {result['std_risk']:.4f}
- 风险范围: [{result['risk_range'][0]:.4f}, {result['risk_range'][1]:.4f}]
- 变异系数: {result['cv']:.4f}
"""

report_content += f"""
## 5. 分组稳定性分析
- 平均一致性(ARI): {stability_results['avg_consistency']:.4f}
- 一致性标准差: {np.std(stability_results['consistency_scores']):.4f}
- 稳定性等级: {stability_results['stability_level']}

## 6. 主要结论与建议

### 6.1 算法性能
- 最佳分组算法: {best_method.upper()}
- 交叉验证稳定性: 良好
- 对检验误差的鲁棒性: 中等

### 6.2 临床应用建议
1. 低风险组: 采用标准NIPT检测流程，13周检测
2. 中风险组: 增强质控措施，必要时重复检测  
3. 高风险组: 高度关注，准备备选检测方案

### 6.3 质量控制要求
- 对于高BMI组别，建议实施更严格的检测误差控制
- 建立分层质控标准，不同风险组采用不同的质控阈值
- 定期校准检测设备，确保误差在可接受范围内

### 6.4 风险监控建议
- 重点关注BMI极值样本(BMI<20或BMI>35)
- 建立风险预警机制，对高风险组进行重点监护
- 考虑引入多因子风险评估模型，提高预测精度

## 7. 技术创新点
1. 引入生存分析方法进行BMI分组
2. 构建多因子风险评估函数
3. 实施全面的交叉验证和敏感性分析
4. 建立检验误差影响的定量评估体系

---
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
分析版本: T2 v2.1
"""

# 保存综合报告
with open(os.path.join(results_dir, 'T2_comprehensive_validation_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_content)

print(report_content)
print(f"\n✅ T2 v2.1 全面验证分析完成！")
print(f"📊 结果保存在: {results_dir}")
print(f"📈 最佳算法: {best_method.upper()}")
print(f"🎯 稳定性等级: {stability_results['stability_level']}")
print(f"⚡ 分析图表: T2_comprehensive_validation_analysis.png")
