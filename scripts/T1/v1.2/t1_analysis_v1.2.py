#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1 分析代码 v1.2
基于清洗后的男性胎儿数据，计算Y染色体浓度与孕周、BMI关系的显著性p值
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_cleaned_data():
    """加载清洗后的数据"""
    print("=== 加载清洗后的数据 ===")
    
    try:
        # 读取清洗后的CSV文件
        # 获取项目根目录路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        df = pd.read_csv(os.path.join(project_root, 'data', 'T1', 'source', 'male_fetal_data_cleaned.csv'))
        print(f"成功加载清洗后的数据，形状: {df.shape}")
        
        # 显示基本信息
        print(f"样本数: {len(df)}")
        print(f"列数: {df.shape[1]}")
        
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def explore_cleaned_data(df):
    """探索清洗后的数据"""
    print("\n=== 清洗后数据探索 ===")
    
    # 关键变量统计
    key_vars = ['检测孕周', '孕妇BMI', 'Y染色体浓度', '年龄', '身高', '体重']
    
    print("关键变量统计:")
    for var in key_vars:
        if var in df.columns:
            stats_info = df[var].describe()
            print(f"\n{var}:")
            print(f"  均值: {stats_info['mean']:.3f}")
            print(f"  标准差: {stats_info['std']:.3f}")
            print(f"  最小值: {stats_info['min']:.3f}")
            print(f"  最大值: {stats_info['max']:.3f}")
            print(f"  中位数: {stats_info['50%']:.3f}")
    
    # 缺失值检查
    print(f"\n缺失值统计:")
    missing = df[key_vars].isnull().sum()
    for var, count in missing.items():
        if count > 0:
            print(f"  {var}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print(f"  {var}: 无缺失值")
    
    return df

def calculate_correlation_significance(df):
    """计算相关性显著性"""
    print("\n=== 相关性显著性分析 ===")
    
    # 关键变量
    ga_col = '检测孕周'
    bmi_col = '孕妇BMI'
    y_col = 'Y染色体浓度'
    age_col = '年龄'
    
    results = {}
    
    # 1. Y染色体浓度与孕周的相关性
    print("1. Y染色体浓度与孕周的相关性分析")
    if ga_col in df.columns and y_col in df.columns:
        # 去除缺失值
        valid_data = df[[ga_col, y_col]].dropna()
        
        if len(valid_data) > 0:
            # Pearson相关系数
            pearson_r, pearson_p = pearsonr(valid_data[ga_col], valid_data[y_col])
            print(f"   Pearson相关系数: r = {pearson_r:.4f}, p = {pearson_p:.6f}")
            
            # Spearman相关系数
            spearman_r, spearman_p = spearmanr(valid_data[ga_col], valid_data[y_col])
            print(f"   Spearman相关系数: ρ = {spearman_r:.4f}, p = {spearman_p:.6f}")
            
            # 显著性判断
            if pearson_p < 0.001:
                significance = "极显著 (p < 0.001)"
            elif pearson_p < 0.01:
                significance = "高度显著 (p < 0.01)"
            elif pearson_p < 0.05:
                significance = "显著 (p < 0.05)"
            else:
                significance = "不显著 (p ≥ 0.05)"
            
            print(f"   显著性: {significance}")
            
            results['ga_y_correlation'] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'significance': significance,
                'n': len(valid_data)
            }
    
    # 2. Y染色体浓度与BMI的相关性
    print("\n2. Y染色体浓度与BMI的相关性分析")
    if bmi_col in df.columns and y_col in df.columns:
        valid_data = df[[bmi_col, y_col]].dropna()
        
        if len(valid_data) > 0:
            # Pearson相关系数
            pearson_r, pearson_p = pearsonr(valid_data[bmi_col], valid_data[y_col])
            print(f"   Pearson相关系数: r = {pearson_r:.4f}, p = {pearson_p:.6f}")
            
            # Spearman相关系数
            spearman_r, spearman_p = spearmanr(valid_data[bmi_col], valid_data[y_col])
            print(f"   Spearman相关系数: ρ = {spearman_r:.4f}, p = {spearman_p:.6f}")
            
            # 显著性判断
            if pearson_p < 0.001:
                significance = "极显著 (p < 0.001)"
            elif pearson_p < 0.01:
                significance = "高度显著 (p < 0.01)"
            elif pearson_p < 0.05:
                significance = "显著 (p < 0.05)"
            else:
                significance = "不显著 (p ≥ 0.05)"
            
            print(f"   显著性: {significance}")
            
            results['bmi_y_correlation'] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'significance': significance,
                'n': len(valid_data)
            }
    
    
    return results

def create_visualizations(df):
    """创建可视化图表"""
    print("\n=== 创建可视化图表 ===")
    
    # 设置图表样式
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Y染色体浓度与孕周的散点图
    plt.subplot(3, 3, 1)
    plt.scatter(df['检测孕周'], df['Y染色体浓度'], alpha=0.6, s=20)
    plt.xlabel('检测孕周 (周)')
    plt.ylabel('Y染色体浓度 (%)')
    plt.title('Y染色体浓度与孕周的关系')
    
    # 添加趋势线
    z = np.polyfit(df['检测孕周'], df['Y染色体浓度'], 1)
    p = np.poly1d(z)
    plt.plot(df['检测孕周'], p(df['检测孕周']), "r--", alpha=0.8)
    
    # 2. Y染色体浓度与BMI的散点图
    plt.subplot(3, 3, 2)
    plt.scatter(df['孕妇BMI'], df['Y染色体浓度'], alpha=0.6, s=20, color='orange')
    plt.xlabel('孕妇BMI (kg/m²)')
    plt.ylabel('Y染色体浓度 (%)')
    plt.title('Y染色体浓度与BMI的关系')
    
    # 添加趋势线
    z = np.polyfit(df['孕妇BMI'], df['Y染色体浓度'], 1)
    p = np.poly1d(z)
    plt.plot(df['孕妇BMI'], p(df['孕妇BMI']), "r--", alpha=0.8)
    
    # 3. BMI分组箱线图
    plt.subplot(3, 3, 3)
    df['bmi_group'] = pd.cut(df['孕妇BMI'], 
                            bins=[0, 18.5, 25, 30, 35, 100], 
                            labels=['偏瘦', '正常', '超重', '肥胖I级', '肥胖II级'])
    
    # 只显示有数据的组
    valid_groups = df['bmi_group'].dropna().unique()
    bmi_data = [df[df['bmi_group'] == group]['Y染色体浓度'].values for group in valid_groups]
    
    plt.boxplot(bmi_data, labels=valid_groups)
    plt.xlabel('BMI分组')
    plt.ylabel('Y染色体浓度 (%)')
    plt.title('不同BMI分组的Y染色体浓度分布')
    plt.xticks(rotation=45)
    
    # 4. 孕周分组箱线图
    plt.subplot(3, 3, 4)
    df['ga_group'] = pd.cut(df['检测孕周'], 
                           bins=[0, 12, 16, 20, 25, 30], 
                           labels=['≤12周', '12-16周', '16-20周', '20-25周', '>25周'])
    
    # 只显示有数据的组
    valid_ga_groups = df['ga_group'].dropna().unique()
    ga_data = [df[df['ga_group'] == group]['Y染色体浓度'].values for group in valid_ga_groups]
    
    plt.boxplot(ga_data, labels=valid_ga_groups)
    plt.xlabel('孕周分组')
    plt.ylabel('Y染色体浓度 (%)')
    plt.title('不同孕周分组的Y染色体浓度分布')
    plt.xticks(rotation=45)
    
    # 5. Y染色体浓度分布直方图
    plt.subplot(3, 3, 5)
    plt.hist(df['Y染色体浓度'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=4, color='red', linestyle='--', label='NIPT失败阈值 (4%)')
    plt.xlabel('Y染色体浓度 (%)')
    plt.ylabel('频数')
    plt.title('Y染色体浓度分布')
    plt.legend()
    
    # 6. 孕周分布直方图
    plt.subplot(3, 3, 6)
    plt.hist(df['检测孕周'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('检测孕周 (周)')
    plt.ylabel('频数')
    plt.title('检测孕周分布')
    
    # 7. BMI分布直方图
    plt.subplot(3, 3, 7)
    plt.hist(df['孕妇BMI'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('孕妇BMI (kg/m²)')
    plt.ylabel('频数')
    plt.title('孕妇BMI分布')
    
    # 8. 孕周与BMI的散点图（颜色表示Y染色体浓度）
    plt.subplot(3, 3, 8)
    scatter = plt.scatter(df['检测孕周'], df['孕妇BMI'], c=df['Y染色体浓度'], 
                         cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Y染色体浓度 (%)')
    plt.xlabel('检测孕周 (周)')
    plt.ylabel('孕妇BMI (kg/m²)')
    plt.title('孕周与BMI关系 (颜色表示Y染色体浓度)')
    
    # 9. NIPT失败率对比
    plt.subplot(3, 3, 9)
    
    # 按BMI分组计算失败率
    bmi_failure_rates = []
    bmi_labels = []
    for group in valid_groups:
        group_data = df[df['bmi_group'] == group]
        if len(group_data) > 0:
            failure_rate = (group_data['Y染色体浓度'] < 4).mean() * 100
            bmi_failure_rates.append(failure_rate)
            bmi_labels.append(f"{group}\n(n={len(group_data)})")
    
    bars = plt.bar(range(len(bmi_failure_rates)), bmi_failure_rates, 
                   color=['lightblue', 'lightgreen', 'orange', 'lightcoral', 'pink'])
    plt.xlabel('BMI分组')
    plt.ylabel('NIPT失败率 (%)')
    plt.title('不同BMI分组的NIPT失败率')
    plt.xticks(range(len(bmi_labels)), bmi_labels, rotation=45)
    
    # 在柱子上添加数值标签
    for i, (bar, rate) in enumerate(zip(bars, bmi_failure_rates)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('t1_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("可视化图表已保存为: t1_analysis_visualizations.png")
    
    return fig

def build_segmented_linear_model(df):
    """构建分段线性回归模型"""
    print("\n=== 分段线性回归模型构建 ===")
    
    # 准备数据
    ga_col = '检测孕周'
    bmi_col = '孕妇BMI'
    y_col = 'Y染色体浓度'
    age_col = '年龄'
    
    # 去除缺失值
    model_data = df[[ga_col, bmi_col, y_col, age_col]].dropna()
    print(f"模型数据样本数: {len(model_data)}")
    
    if len(model_data) < 10:
        print("样本数不足，无法构建模型")
        return None
    
    # 创建分段变量
    # 孕周分段：21周为临界值
    model_data['ga_phase1'] = model_data[ga_col]  # 基础孕周
    model_data['ga_phase2'] = np.where(model_data[ga_col] >= 21, 
                                      model_data[ga_col] - 21, 0)  # 21周后增量
    
    # BMI分段：35 kg/m²为临界值
    model_data['bmi_phase1'] = model_data[bmi_col]  # 基础BMI
    model_data['bmi_phase2'] = np.where(model_data[bmi_col] >= 35, 
                                       model_data[bmi_col] - 35, 0)  # 35以上增量
    
    # 交互项
    model_data['ga_bmi_interaction'] = model_data[ga_col] * model_data[bmi_col]
    model_data['age_bmi_interaction'] = model_data[age_col] * model_data[bmi_col]
    
    # 构建模型
    X = model_data[['ga_phase1', 'ga_phase2', 'bmi_phase1', 'bmi_phase2', 
                   'ga_bmi_interaction', 'age_bmi_interaction']]
    y = model_data[y_col]
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 线性回归
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # 预测
    y_pred = model.predict(X_scaled)
    
    # 模型评估
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"模型R²: {r2:.4f}")
    print(f"模型RMSE: {rmse:.4f}")
    
    # 计算系数显著性（使用t检验）
    n = len(y)
    p = X_scaled.shape[1]
    mse = mean_squared_error(y, y_pred)
    
    # 计算标准误
    XTX_inv = np.linalg.inv(X_scaled.T @ X_scaled)
    se = np.sqrt(np.diag(XTX_inv) * mse)
    
    # t统计量和p值
    t_stats = model.coef_ / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
    
    # 系数解释
    feature_names = ['孕周基础', '孕周21周后增量', 'BMI基础', 'BMI35以上增量', 
                    '孕周×BMI交互', '年龄×BMI交互']
    
    print("\n模型系数显著性:")
    for i, (name, coef, se_val, t_stat, p_val) in enumerate(zip(feature_names, model.coef_, se, t_stats, p_values)):
        if p_val < 0.001:
            significance = "极显著 (p < 0.001)"
        elif p_val < 0.01:
            significance = "高度显著 (p < 0.01)"
        elif p_val < 0.05:
            significance = "显著 (p < 0.05)"
        else:
            significance = "不显著 (p ≥ 0.05)"
        
        print(f"  {name}: 系数={coef:.4f}, 标准误={se_val:.4f}, t={t_stat:.4f}, p={p_val:.6f} ({significance})")
    
    # 截距
    intercept_se = np.sqrt(mse * (1/n + np.mean(X_scaled, axis=0) @ XTX_inv @ np.mean(X_scaled, axis=0)))
    intercept_t = model.intercept_ / intercept_se
    intercept_p = 2 * (1 - stats.t.cdf(np.abs(intercept_t), n - p - 1))
    
    if intercept_p < 0.001:
        significance = "极显著 (p < 0.001)"
    elif intercept_p < 0.01:
        significance = "高度显著 (p < 0.01)"
    elif intercept_p < 0.05:
        significance = "显著 (p < 0.05)"
    else:
        significance = "不显著 (p ≥ 0.05)"
    
    print(f"  截距: 系数={model.intercept_:.4f}, 标准误={intercept_se:.4f}, t={intercept_t:.4f}, p={intercept_p:.6f} ({significance})")
    
    return {
        'model': model,
        'scaler': scaler,
        'r2': r2,
        'rmse': rmse,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'p_values': p_values,
        'intercept_p': intercept_p,
        'feature_names': feature_names,
        'n': n
    }

def bmi_group_analysis(df):
    """BMI分组分析"""
    print("\n=== BMI分组分析 ===")
    
    # BMI分组
    df['bmi_group'] = pd.cut(df['孕妇BMI'], 
                            bins=[0, 18.5, 25, 30, 35, 100], 
                            labels=['偏瘦', '正常', '超重', '肥胖I级', '肥胖II级'])
    
    # 各组统计
    group_stats = df.groupby('bmi_group').agg({
        'Y染色体浓度': ['count', 'mean', 'std', 'min', 'max'],
        '检测孕周': ['mean', 'std'],
        '年龄': ['mean', 'std']
    }).round(3)
    
    print("BMI分组统计:")
    print(group_stats)
    
    # 各组Y染色体浓度比较
    print("\n各组Y染色体浓度比较:")
    groups = []
    for group in df['bmi_group'].cat.categories:
        group_data = df[df['bmi_group'] == group]['Y染色体浓度'].dropna()
        if len(group_data) > 0:
            groups.append(group_data)
            print(f"  {group}: n={len(group_data)}, 均值={group_data.mean():.3f}%, 标准差={group_data.std():.3f}%")
    
    # ANOVA检验
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"\nANOVA检验: F={f_stat:.4f}, p={p_value:.6f}")
        
        if p_value < 0.001:
            significance = "极显著 (p < 0.001)"
        elif p_value < 0.01:
            significance = "高度显著 (p < 0.01)"
        elif p_value < 0.05:
            significance = "显著 (p < 0.05)"
        else:
            significance = "不显著 (p ≥ 0.05)"
        
        print(f"组间差异: {significance}")
    
    return group_stats

def gestational_age_analysis(df):
    """孕周分析"""
    print("\n=== 孕周分析 ===")
    
    # 孕周分组
    df['ga_group'] = pd.cut(df['检测孕周'], 
                           bins=[0, 12, 16, 20, 25, 30], 
                           labels=['早期(≤12周)', '早中期(12-16周)', '中期(16-20周)', '中晚期(20-25周)', '晚期(>25周)'])
    
    # 各组统计
    group_stats = df.groupby('ga_group').agg({
        'Y染色体浓度': ['count', 'mean', 'std', 'min', 'max'],
        '孕妇BMI': ['mean', 'std'],
        '年龄': ['mean', 'std']
    }).round(3)
    
    print("孕周分组统计:")
    print(group_stats)
    
    # 各组Y染色体浓度比较
    print("\n各组Y染色体浓度比较:")
    groups = []
    for group in df['ga_group'].cat.categories:
        group_data = df[df['ga_group'] == group]['Y染色体浓度'].dropna()
        if len(group_data) > 0:
            groups.append(group_data)
            print(f"  {group}: n={len(group_data)}, 均值={group_data.mean():.3f}%, 标准差={group_data.std():.3f}%")
    
    # ANOVA检验
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"\nANOVA检验: F={f_stat:.4f}, p={p_value:.6f}")
        
        if p_value < 0.001:
            significance = "极显著 (p < 0.001)"
        elif p_value < 0.01:
            significance = "高度显著 (p < 0.01)"
        elif p_value < 0.05:
            significance = "显著 (p < 0.05)"
        else:
            significance = "不显著 (p ≥ 0.05)"
        
        print(f"组间差异: {significance}")
    
    return group_stats

def clinical_threshold_analysis(df):
    """临床阈值分析"""
    print("\n=== 临床阈值分析 ===")
    
    # NIPT检测失败阈值（Y染色体浓度 < 4%）
    df['nipt_failure'] = df['Y染色体浓度'] < 4.0
    
    failure_rate = df['nipt_failure'].mean() * 100
    print(f"NIPT检测失败率 (Y染色体浓度 < 4%): {failure_rate:.1f}% ({df['nipt_failure'].sum()}例)")
    
    # 按BMI分组分析失败率
    print("\n按BMI分组的检测失败率:")
    bmi_groups = pd.cut(df['孕妇BMI'], 
                       bins=[0, 25, 30, 35, 100], 
                       labels=['正常/偏瘦', '超重', '肥胖I级', '肥胖II级'])
    
    for group in bmi_groups.cat.categories:
        group_data = df[bmi_groups == group]
        if len(group_data) > 0:
            group_failure_rate = group_data['nipt_failure'].mean() * 100
            print(f"  {group}: {group_failure_rate:.1f}% ({group_data['nipt_failure'].sum()}/{len(group_data)}例)")
    
    # 按孕周分组分析失败率
    print("\n按孕周分组的检测失败率:")
    ga_groups = pd.cut(df['检测孕周'], 
                      bins=[0, 12, 16, 20, 25, 30], 
                      labels=['≤12周', '12-16周', '16-20周', '20-25周', '>25周'])
    
    for group in ga_groups.cat.categories:
        group_data = df[ga_groups == group]
        if len(group_data) > 0:
            group_failure_rate = group_data['nipt_failure'].mean() * 100
            print(f"  {group}: {group_failure_rate:.1f}% ({group_data['nipt_failure'].sum()}/{len(group_data)}例)")
    
    return {
        'overall_failure_rate': failure_rate,
        'bmi_failure_rates': df.groupby(bmi_groups)['nipt_failure'].agg(['count', 'sum', 'mean']),
        'ga_failure_rates': df.groupby(ga_groups)['nipt_failure'].agg(['count', 'sum', 'mean'])
    }

def generate_summary_report(correlation_results, model_results, bmi_stats, ga_stats, clinical_results):
    """生成总结报告"""
    print("\n" + "="*60)
    print("T1 分析总结报告")
    print("="*60)
    
    print("\n1. 相关性分析结果:")
    if 'ga_y_correlation' in correlation_results:
        result = correlation_results['ga_y_correlation']
        print(f"   Y染色体浓度与孕周: r={result['pearson_r']:.4f}, p={result['pearson_p']:.6f} ({result['significance']})")
    
    if 'bmi_y_correlation' in correlation_results:
        result = correlation_results['bmi_y_correlation']
        print(f"   Y染色体浓度与BMI: r={result['pearson_r']:.4f}, p={result['pearson_p']:.6f} ({result['significance']})")
    
    
    print("\n2. 分段线性回归模型:")
    if model_results:
        print(f"   模型R²: {model_results['r2']:.4f}")
        print(f"   模型RMSE: {model_results['rmse']:.4f}")
        print("   显著变量:")
        for i, (name, p_val) in enumerate(zip(model_results['feature_names'], model_results['p_values'])):
            if p_val < 0.05:
                print(f"     {name}: p={p_val:.6f}")
    
    print("\n3. 临床意义:")
    if clinical_results:
        print(f"   总体NIPT失败率: {clinical_results['overall_failure_rate']:.1f}%")
        print("   按BMI分组失败率:")
        for group, rates in clinical_results['bmi_failure_rates'].iterrows():
            if pd.notna(group):
                print(f"     {group}: {rates['mean']*100:.1f}%")
    
    print("\n4. 建模建议:")
    print("   基于显著性分析结果，建议:")
    if 'ga_y_correlation' in correlation_results and correlation_results['ga_y_correlation']['pearson_p'] < 0.05:
        print("   - 孕周与Y染色体浓度显著相关，应纳入模型")
    if 'bmi_y_correlation' in correlation_results and correlation_results['bmi_y_correlation']['pearson_p'] < 0.05:
        print("   - BMI与Y染色体浓度显著相关，应纳入模型")
    if model_results and any(p < 0.05 for p in model_results['p_values']):
        print("   - 分段线性模型中的显著变量可用于预测Y染色体浓度")
    
    print("\n分析完成！")

def main():
    """主函数"""
    print("T1 分析代码 v1.2")
    print("基于清洗后的男性胎儿数据计算显著性p值")
    print("="*60)
    
    # 1. 加载数据
    df = load_cleaned_data()
    if df is None:
        return
    
    # 2. 探索数据
    df = explore_cleaned_data(df)
    
    # 3. 相关性显著性分析
    correlation_results = calculate_correlation_significance(df)
    
    # 4. 创建可视化图表
    create_visualizations(df)
    
    # 5. 构建分段线性模型
    model_results = build_segmented_linear_model(df)
    
    # 6. BMI分组分析
    bmi_stats = bmi_group_analysis(df)
    
    # 7. 孕周分析
    ga_stats = gestational_age_analysis(df)
    
    # 8. 临床阈值分析
    clinical_results = clinical_threshold_analysis(df)
    
    # 9. 生成总结报告
    generate_summary_report(correlation_results, model_results, bmi_stats, ga_stats, clinical_results)
    
    # 10. 保存结果
    print("\n保存分析结果...")
    results_df = pd.DataFrame({
        '分析项目': ['Y染色体浓度-孕周相关性', 'Y染色体浓度-BMI相关性'],
        'Pearson相关系数': [
            correlation_results.get('ga_y_correlation', {}).get('pearson_r', np.nan),
            correlation_results.get('bmi_y_correlation', {}).get('pearson_r', np.nan)
        ],
        'P值': [
            correlation_results.get('ga_y_correlation', {}).get('pearson_p', np.nan),
            correlation_results.get('bmi_y_correlation', {}).get('pearson_p', np.nan)
        ],
        '显著性': [
            correlation_results.get('ga_y_correlation', {}).get('significance', ''),
            correlation_results.get('bmi_y_correlation', {}).get('significance', '')
        ]
    })
    
    results_df.to_csv('t1_analysis_results.csv', index=False, encoding='utf-8-sig')
    print("分析结果已保存到: t1_analysis_results.csv")

if __name__ == "__main__":
    main()
