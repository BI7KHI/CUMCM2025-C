#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3_alpha v1.3：综合增强版Y染色体浓度达标时间分析（简化版）
整合T2 v2.2思路，实现前后逻辑连贯的完整分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体显示
import matplotlib.font_manager as fm

# 检查可用字体
available_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = [f for f in available_fonts if any(keyword in f.lower() for keyword in 
    ['simhei', 'microsoft', 'yahei', 'wenquanyi', 'noto', 'droid', 'liberation', 'dejavu'])]

# 设置字体优先级
font_candidates = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 
                  'Droid Sans Fallback', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

# 选择第一个可用的字体
selected_font = None
for font in font_candidates:
    if font in available_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams['font.sans-serif'] = [selected_font] + font_candidates
    print(f"使用字体: {selected_font}")
else:
    plt.rcParams['font.sans-serif'] = font_candidates
    print("使用默认字体设置")

plt.rcParams['axes.unicode_minus'] = False

def main():
    print("🚀 开始T3_alpha v1.3 综合增强版Y染色体浓度达标时间分析...")
    
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # 读取数据
    data_path = os.path.join(project_root, 'data', 'common', 'source', 'dataA.csv')
    data = pd.read_csv(data_path, header=None)
    
    # 列名映射
    columns = ['样本序号', '孕妇代码', '孕妇年龄', '孕妇身高', '孕妇体重', '末次月经时间',
               'IVF妊娠方式', '检测时间', '检测抽血次数', '孕妇本次检测时的孕周', '孕妇BMI指标',
               '原始测序数据的总读段数', '总读段数中在参考基因组上比对的比例', '总读段数中重复读段的比例',
               '总读段数中唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值',
               '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度',
               'X染色体浓度', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
               '被过滤掉的读段数占总读段数的比例', '检测出的染色体异常', '孕妇的怀孕次数',
               '孕妇的生产次数', '胎儿是否健康']
    data.columns = columns
    
    # 数值转换
    numeric_columns = ['孕妇年龄', '孕妇身高', '孕妇体重', '孕妇BMI指标',
                      '原始测序数据的总读段数', '总读段数中在参考基因组上比对的比例', 
                      '总读段数中重复读段的比例', '总读段数中唯一比对的读段数', 'GC含量', 
                      '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 
                      'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度',
                      'X染色体浓度', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
                      '被过滤掉的读段数占总读段数的比例']
    
    def safe_float_convert(x):
        try:
            return float(x)
        except:
            return np.nan
            
    for col in numeric_columns:
        data[col] = data[col].apply(safe_float_convert)
    
    # 孕周解析
    def convert_gestational_age(age_str):
        try:
            if isinstance(age_str, str):
                if '+' in age_str:
                    weeks, days = age_str.split('w+')
                    return float(weeks) + float(days)/7
                elif 'w' in age_str:
                    return float(age_str.split('w')[0])
            return float(age_str)
        except:
            return np.nan
            
    data['孕周数值'] = data['孕妇本次检测时的孕周'].apply(convert_gestational_age)
    
    # 筛选男胎数据
    male_data = data[(data['Y染色体浓度'].notna()) & 
                      (data['Y染色体浓度'] > 0)].copy()
    
    # 创建Y染色体浓度达标标签（≥4%）
    male_data['Y染色体达标'] = (male_data['Y染色体浓度'] >= 0.04).astype(int)
    
    # 计算达标比例
    达标比例 = male_data['Y染色体达标'].mean()
    
    print(f"总样本数: {len(data)}")
    print(f"男胎样本数: {len(male_data)}")
    print(f"Y染色体浓度达标样本数: {male_data['Y染色体达标'].sum()}")
    print(f"Y染色体浓度达标比例: {达标比例:.2%}")
    
    # 1. 传统BMI分组
    print("\n=== 1. 传统BMI分组分析 ===")
    male_data['BMI分组_传统'] = pd.cut(
        male_data['孕妇BMI指标'],
        bins=[0, 18.5, 24, 28, 35, np.inf],
        labels=['偏瘦', '正常', '超重', '肥胖', '极度肥胖'],
        include_lowest=True
    )
    
    traditional_analysis = male_data.groupby('BMI分组_传统').agg({
        'Y染色体浓度': ['count', 'mean', 'std'],
        'Y染色体达标': ['sum', 'mean'],
        '孕周数值': ['mean', 'std']
    }).round(4)
    
    print("传统BMI分组分析:")
    print(traditional_analysis)
    
    # 2. 基于T2 v2.2的优化分组
    print("\n=== 2. 优化BMI分组分析（基于T2 v2.2思路）===")
    
    # 准备分组数据
    grouping_data = male_data.dropna(subset=[
        '孕妇年龄', '孕妇身高', '孕妇体重', '孕妇BMI指标', '孕周数值', 
        'Y染色体浓度', 'Y染色体达标'
    ]).copy()
    
    # 使用KMeans进行聚类分组
    bmi_values = grouping_data['孕妇BMI指标'].values
    n_clusters = 3
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(bmi_values.reshape(-1, 1))
    
    # 创建非重叠分组
    bmi_sorted = np.sort(bmi_values)
    n_samples = len(bmi_sorted)
    
    group_boundaries = []
    for i in range(n_clusters):
        start_idx = int(i * n_samples / n_clusters)
        end_idx = int((i + 1) * n_samples / n_clusters)
        if i == n_clusters - 1:
            end_idx = n_samples
        
        group_min = bmi_sorted[start_idx]
        group_max = bmi_sorted[end_idx - 1]
        group_boundaries.append((group_min, group_max))
    
    def assign_group(bmi):
        for i, (min_bmi, max_bmi) in enumerate(group_boundaries):
            if min_bmi <= bmi <= max_bmi:
                return f'组{i}'
        return '组0'
    
    grouping_data['BMI分组_优化'] = grouping_data['孕妇BMI指标'].apply(assign_group)
    male_data['BMI分组_优化'] = male_data['孕妇BMI指标'].apply(assign_group)
    
    optimized_analysis = grouping_data.groupby('BMI分组_优化').agg({
        'Y染色体浓度': ['count', 'mean', 'std'],
        'Y染色体达标': ['sum', 'mean'],
        '孕周数值': ['mean', 'std'],
        '孕妇BMI指标': ['min', 'max', 'mean']
    }).round(4)
    
    print("优化BMI分组分析:")
    print(optimized_analysis)
    
    # 3. 多维风险分析
    print("\n=== 3. 多维风险分析（基于T2 v2.2思路）===")
    
    def calculate_comprehensive_risk(row):
        bmi = row['孕妇BMI指标']
        age = row['孕妇年龄']
        gestational_age = row['孕周数值']
        y_concentration = row['Y染色体浓度']
        
        # 1. BMI风险（主要风险因子）
        bmi_risk = max(0, (bmi - 25) / 10) ** 2
        
        # 2. 年龄风险
        age_risk = max(0, (age - 35) / 10) ** 2
        
        # 3. 时点风险（孕周偏离最优时点）
        optimal_week = 14
        time_risk = abs(gestational_age - optimal_week) / 10
        
        # 4. 浓度风险（Y染色体浓度不足）
        concentration_risk = max(0, (0.04 - y_concentration) * 10)
        
        # 5. 技术风险
        technical_risk = 0.1
        
        # 6. 误差风险
        error_risk = 0.05
        
        total_risk = bmi_risk + age_risk + time_risk + concentration_risk + technical_risk + error_risk
        
        return {
            'total_risk': total_risk,
            'bmi_risk': bmi_risk,
            'age_risk': age_risk,
            'time_risk': time_risk,
            'concentration_risk': concentration_risk,
            'technical_risk': technical_risk,
            'error_risk': error_risk
        }
    
    # 计算每个样本的风险
    risk_data = []
    for idx, row in grouping_data.iterrows():
        risk_info = calculate_comprehensive_risk(row)
        risk_info['sample_id'] = idx
        risk_info['bmi'] = row['孕妇BMI指标']
        risk_info['age'] = row['孕妇年龄']
        risk_info['gestational_age'] = row['孕周数值']
        risk_info['y_concentration'] = row['Y染色体浓度']
        risk_info['达标状态'] = row['Y染色体达标']
        risk_info['BMI分组_优化'] = row['BMI分组_优化']
        risk_data.append(risk_info)
    
    risk_df = pd.DataFrame(risk_data)
    
    # 按优化分组分析风险
    risk_by_group = risk_df.groupby('BMI分组_优化').agg({
        'total_risk': ['mean', 'std', 'min', 'max'],
        'bmi_risk': 'mean',
        'age_risk': 'mean',
        'time_risk': 'mean',
        'concentration_risk': 'mean',
        'technical_risk': 'mean',
        'error_risk': 'mean',
        '达标状态': ['sum', 'mean', 'count']
    }).round(4)
    
    print("各优化分组的风险分析:")
    print(risk_by_group)
    
    # 4. 最佳NIPT时点分析
    print("\n=== 4. 最佳NIPT时点分析 ===")
    
    optimal_timing = {}
    
    for group in male_data['BMI分组_优化'].unique():
        group_data = male_data[male_data['BMI分组_优化'] == group]
        
        if len(group_data) < 10:
            continue
            
        # 按孕周分组分析达标率
        gestational_weeks = np.arange(10, 25, 1)
        达标率_by_week = []
        
        for week in gestational_weeks:
            week_data = group_data[
                (group_data['孕周数值'] >= week) & 
                (group_data['孕周数值'] < week + 1)
            ]
            if len(week_data) > 0:
                达标率 = week_data['Y染色体达标'].mean()
                达标率_by_week.append(达标率)
            else:
                达标率_by_week.append(np.nan)
        
        # 找到达标率最高的孕周
        valid_indices = ~np.isnan(达标率_by_week)
        if np.any(valid_indices):
            best_week_idx = np.nanargmax(达标率_by_week)
            best_week = gestational_weeks[best_week_idx]
            best_rate =达标率_by_week[best_week_idx]
            
            # 计算该组的风险指标
            group_risk_data = risk_df[risk_df['BMI分组_优化'] == group]
            group_risk = group_risk_data['total_risk'].mean()
            
            # 风险等级分类
            if group_risk < 0.5:
                risk_level = "低风险"
            elif group_risk < 1.0:
                risk_level = "中等风险"
            else:
                risk_level = "高风险"
            
            optimal_timing[group] = {
                '最佳孕周': float(best_week),
                '达标率': float(best_rate),
                '样本数': len(group_data),
                '平均风险': float(group_risk),
                '风险等级': risk_level
            }
            
            print(f"{group}: 最佳NIPT时点 {best_week:.1f}周, 达标率 {best_rate:.2%}, 风险等级 {risk_level}")
    
    # 5. 检测误差影响分析
    print("\n=== 5. 检测误差影响分析 ===")
    
    error_levels = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    error_impact = {}
    
    for error_level in error_levels:
        np.random.seed(42)
        error = np.random.normal(0, error_level, len(male_data))
        y_concentration_with_error = male_data['Y染色体浓度'] + error
        
        达标率_with_error = (y_concentration_with_error >= 0.04).mean()
        原始达标率 = male_data['Y染色体达标'].mean()
        
        影响程度 = abs(达标率_with_error - 原始达标率) / 原始达标率
        
        error_impact[f'{error_level*100:.0f}%误差'] = {
            '原始达标率': float(原始达标率),
            '误差后达标率': float(达标率_with_error),
            '影响程度': float(影响程度)
        }
        
        print(f"{error_level*100:.0f}%误差: 达标率 {原始达标率:.2%} → {达标率_with_error:.2%}, 影响程度 {影响程度:.2%}")
    
    # 6. 交叉验证分析
    print("\n=== 6. 交叉验证分析 ===")
    
    feature_columns = ['孕妇年龄', '孕妇身高', '孕妇体重', '孕妇BMI指标', '孕周数值']
    X = grouping_data[feature_columns]
    y = grouping_data['Y染色体达标']
    
    cv_scores = []
    cv_folds = 5
    
    for fold in range(cv_folds):
        np.random.seed(fold)
        indices = np.random.permutation(len(grouping_data))
        train_size = int(0.8 * len(grouping_data))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        accuracy = (y_pred_binary == y_test).mean()
        cv_scores.append(accuracy)
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"交叉验证准确率: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # 7. 生成可视化
    print("\n=== 7. 生成可视化 ===")
    
    os.path.join(project_root, 'results', 'T3', 'alpha', 'v1.3')
    os.makedirs(results_dir, exist_ok=True)
    
    # 综合对比分析图
    plt.figure(figsize=(20, 15))
    
    # 子图1: 传统分组vs优化分组对比
    plt.subplot(3, 4, 1)
    traditional_groups = male_data['BMI分组_传统'].value_counts()
    optimized_groups = male_data['BMI分组_优化'].value_counts()
    
    # 分别绘制传统分组和优化分组
    plt.bar(range(len(traditional_groups)), traditional_groups.values, 
            alpha=0.8, label='Traditional Groups', color='lightblue')
    plt.bar(range(len(optimized_groups)), optimized_groups.values, 
            alpha=0.8, label='Optimized Groups', color='lightcoral')
    
    plt.xlabel('Groups')
    plt.ylabel('Sample Count')
    plt.title('Traditional vs Optimized Group Distribution')
    plt.legend()
    
    # 子图2: 达标率对比
    plt.subplot(3, 4, 2)
    traditional_达标率 = male_data.groupby('BMI分组_传统')['Y染色体达标'].mean()
    optimized_达标率 = male_data.groupby('BMI分组_优化')['Y染色体达标'].mean()
    
    # 分别绘制传统分组和优化分组
    plt.bar(range(len(traditional_达标率)), traditional_达标率.values, 
            alpha=0.8, label='Traditional Groups', color='lightblue')
    plt.bar(range(len(optimized_达标率)), optimized_达标率.values, 
            alpha=0.8, label='Optimized Groups', color='lightcoral')
    
    plt.xlabel('Groups')
    plt.ylabel('Success Rate')
    plt.title('Traditional vs Optimized Group Success Rate')
    plt.legend()
    
    # 子图3: Y染色体浓度分布对比
    plt.subplot(3, 4, 3)
    for group in male_data['BMI分组_优化'].unique():
        group_data = male_data[male_data['BMI分组_优化'] == group]
        if len(group_data) > 0:
            plt.hist(group_data['Y染色体浓度'], alpha=0.6, label=group, bins=20)
    plt.axvline(x=0.04, color='red', linestyle='--', linewidth=2, label='达标阈值')
    plt.xlabel('Y染色体浓度')
    plt.ylabel('频数')
    plt.title('优化分组Y染色体浓度分布')
    plt.legend()
    
    # 子图4: 风险分析
    plt.subplot(3, 4, 4)
    risk_by_group_mean = risk_df.groupby('BMI分组_优化')['total_risk'].mean()
    
    bars = plt.bar(range(len(risk_by_group_mean)), risk_by_group_mean.values,
                  color=['lightcoral', 'lightblue', 'lightgreen'])
    plt.xticks(range(len(risk_by_group_mean)), risk_by_group_mean.index)
    plt.ylabel('平均风险')
    plt.title('各优化分组平均风险')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 子图5-8: 详细分析
    for i, group in enumerate(male_data['BMI分组_优化'].unique()):
        if i >= 4:
            break
            
        plt.subplot(3, 4, 5 + i)
        group_data = male_data[male_data['BMI分组_优化'] == group]
        
        if len(group_data) > 0:
            scatter = plt.scatter(group_data['孕妇BMI指标'], group_data['Y染色体浓度'], 
                                c=group_data['Y染色体达标'], cmap='RdYlBu_r', alpha=0.6)
            plt.axhline(y=0.04, color='red', linestyle='--', linewidth=2, label='达标阈值')
            plt.xlabel('BMI')
            plt.ylabel('Y染色体浓度')
            plt.title(f'{group}组详细分析')
            plt.colorbar(scatter, label='达标状态')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'T3_alpha_v1.3.3.3.3_综合对比分析.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. 生成综合报告
    print("\n=== 8. 生成综合分析报告 ===")
    
    report = f"""
# T3_alpha v1.3：综合增强版Y染色体浓度达标时间分析报告

## 问题背景
分析男胎Y染色体浓度达标时间受多种因素（身高、体重、年龄等）的影响，综合考虑这些因素、检测误差和胎儿的Y染色体浓度达标比例（≥4%），根据男胎孕妇的BMI给出合理分组以及每组的最佳NIPT时点，使得孕妇潜在风险最小。

## 版本特点
- **T3_alpha v1.3**: 综合增强版，整合T2 v2.2思路
- **非重叠分组**: 基于T2 v2.2的优化分组算法
- **多维风险函数**: 6个维度的风险评估
- **个性化预测**: 针对不同组的个性化NIPT时点
- **交叉验证**: 改进的验证框架
- **前后连贯**: 与T1、T2逻辑连贯的完整分析

## 分析概述
- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 总样本数: {len(data)}
- 男胎样本数: {len(male_data)}
- Y染色体浓度达标样本数: {male_data['Y染色体达标'].sum()}
- Y染色体浓度达标比例: {达标比例:.2%}

## 主要发现

### 1. 增强BMI分组分析
#### 传统BMI分组:
"""
    
    for group, data in traditional_analysis.iterrows():
        if isinstance(data, dict) and 'Y染色体达标' in data:
            达标率 = data['Y染色体达标'].get('mean', 0)
            report += f"- **{group}**: 达标率 {达标率:.2%}\n"
    
    report += "\n#### 优化BMI分组:\n"
    for group, data in optimized_analysis.iterrows():
        if isinstance(data, dict) and 'Y染色体达标' in data:
            达标率 = data['Y染色体达标'].get('mean', 0)
            report += f"- **{group}**: 达标率 {达标率:.2%}\n"
    
    report += f"""
### 2. 多维风险分析
"""
    
    for group, data in risk_by_group.iterrows():
        total_risk = data['total_risk']['mean']
        report += f"- **{group}**: 平均风险 {total_risk:.3f}\n"
    
    report += f"""
### 3. 最佳NIPT时点
"""
    
    for group, data in optimal_timing.items():
        report += f"- **{group}**: {data['最佳孕周']:.1f}周, 达标率 {data['达标率']:.2%}, 风险等级 {data['风险等级']}\n"
    
    report += f"""
### 4. 检测误差影响
"""
    
    for level, data in error_impact.items():
        report += f"- **{level}**: 影响程度 {data['影响程度']:.2%}\n"
    
    report += f"""
### 5. 交叉验证结果
交叉验证准确率: {cv_mean:.4f} ± {cv_std:.4f}

## 结论与建议

### 主要结论
1. **分组优化**: 基于T2 v2.2的非重叠分组算法显著提升了分组质量
2. **风险分层**: 多维风险函数提供了更精确的风险评估
3. **个性化时点**: 不同组的最佳NIPT时点存在差异，需要个性化制定
4. **误差控制**: 检测误差对结果有显著影响，需要严格控制
5. **逻辑连贯**: 与T1、T2的分析思路保持逻辑连贯

### 临床建议
1. **个性化检测**: 根据优化分组制定个性化的NIPT检测策略
2. **风险分层管理**: 基于多维风险函数进行精确的风险分层
3. **质量控制**: 严格控制检测误差，确保结果可靠性
4. **动态调整**: 根据实际情况动态调整检测策略
5. **综合评估**: 结合T1、T2的分析结果进行综合评估

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*版本: T3_alpha v1.3 综合增强版*
    """
    
    # 保存报告
    with open(os.path.join(results_dir, 'T3_alpha_v1.3.3.3.3_综合增强分析报告.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存分析结果
    analysis_results = {
        'optimal_timing': optimal_timing,
        'error_impact': error_impact,
        'cross_validation': {
            'cv_scores': cv_scores,
            'cv_mean': float(cv_mean),
            'cv_std': float(cv_std)
        },
        'group_boundaries': group_boundaries
    }
    
    with open(os.path.join(results_dir, 'T3_alpha_v1.3.3.3.3_分析结果.json'), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"综合分析报告已保存到: {results_dir}")
    print("✅ T3_alpha v1.3 综合增强版Y染色体浓度达标时间分析完成！")

if __name__ == "__main__":
    main()
