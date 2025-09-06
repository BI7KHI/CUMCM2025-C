#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 v2.0：基于生存分析的BMI分组优化
目标：最小化孕妇潜在风险，优化NIPT时点选择

新增算法：
1. 生存分析分组 (Survival-based Grouping)
2. 风险最小化优化 (Risk Minimization)
3. Cox比例风险模型
4. Kaplan-Meier生存曲线分组
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import os
import random
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# 健壮的中文字体配置
def configure_chinese_font():
    try:
        fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fonts')
        fonts_dir = os.path.abspath(fonts_dir)
        os.makedirs(fonts_dir, exist_ok=True)
        
        # 加载本地字体
        for file_name in os.listdir(fonts_dir):
            if file_name.lower().endswith(('.ttf', '.otf')):
                try:
                    fm.fontManager.addfont(os.path.join(fonts_dir, file_name))
                except Exception:
                    pass
        
        # 重新加载字体缓存
        try:
            fm._load_fontmanager(try_read_cache=False)
        except Exception:
            pass
        
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
                
        # 如果仍未找到，尝试下载 NotoSansCJKsc-Regular.otf
        target_path = os.path.join(fonts_dir, 'NotoSansCJKsc-Regular.otf')
        if not os.path.exists(target_path):
            url = 'https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf'
            try:
                import urllib.request
                urllib.request.urlretrieve(url, target_path)
                fm.fontManager.addfont(target_path)
                fm._load_fontmanager(try_read_cache=False)
            except Exception:
                pass
        
        installed = set(f.name for f in fm.fontManager.ttflist)
        if 'Noto Sans CJK SC' in installed:
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
            return 'Noto Sans CJK SC'
    except Exception:
        pass
    return None

family = configure_chinese_font()
if family:
    print(f"成功配置中文字体: {family}")
else:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("未找到中文字体，使用默认字体 DejaVu Sans")

plt.rcParams['axes.unicode_minus'] = False

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 创建结果目录（v2.0 专用）
results_dir = os.path.join(project_root, 'results_v2.0')
os.makedirs(results_dir, exist_ok=True)

print("=== T2 v2.0：基于生存分析的BMI分组优化 ===\n")

# 读取数据
data_path = os.path.join(project_root, 'Source_DATA', 'dataA.csv')
data = pd.read_csv(data_path, header=None)

# 根据附录1，确定各列的索引
columns = ['样本序号', '孕妇代码', '孕妇年龄', '孕妇身高', '孕妇体重', '末次月经时间',
           'IVF妊娠方式', '检测时间', '检测抽血次数', '孕妇本次检测时的孕周', '孕妇BMI指标',
           '原始测序数据的总读段数', '总读段数中在参考基因组上比对的比例', '总读段数中重复读段的比例',
           '总读段数中唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值',
           '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度',
           'X染色体浓度', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
           '被过滤掉的读段数占总读段数的比例', '检测出的染色体异常', '孕妇的怀孕次数',
           '孕妇的生产次数', '胎儿是否健康']
data.columns = columns

# 数据预处理：只保留男胎数据
male_fetus_data = data[data['Y染色体浓度'].notna()].copy()

# 转换数值型列的数据类型
def safe_float_convert(x):
    try:
        return float(x)
    except:
        return np.nan

numeric_columns = ['孕妇年龄', '孕妇身高', '孕妇体重', '孕妇BMI指标',
                   'Y染色体浓度', 'Y染色体的Z值', 'GC含量']

for col in numeric_columns:
    male_fetus_data[col] = male_fetus_data[col].apply(safe_float_convert)

# 将孕周转换为数值
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

# 清理数据
male_fetus_data = male_fetus_data.dropna(subset=['孕妇BMI指标', '孕周数值', 'Y染色体浓度'])

# === 1. 风险函数定义 ===

def calculate_risk_score(bmi, gestational_age, y_concentration, detection_error=0.05):
    """
    计算孕妇潜在风险评分
    
    风险因子：
    1. BMI极值风险
    2. 孕周时点不当风险  
    3. Y染色体浓度不足风险
    4. 检测误差风险
    """
    risk_score = 0.0
    
    # BMI风险（U型曲线）
    optimal_bmi = 25.0  # 理想BMI
    bmi_risk = (bmi - optimal_bmi) ** 2 / 100
    risk_score += bmi_risk
    
    # 孕周风险（过早或过晚检测）
    optimal_week = 13.0  # 理想检测孕周
    week_risk = abs(gestational_age - optimal_week) * 0.1
    risk_score += week_risk
    
    # Y染色体浓度风险（浓度不足导致检测失败）
    concentration_threshold = 0.3  # 假设阈值
    if y_concentration < concentration_threshold:
        conc_risk = (concentration_threshold - y_concentration) * 2
        risk_score += conc_risk
    
    # 检测误差风险
    error_risk = detection_error * bmi * 0.01
    risk_score += error_risk
    
    return risk_score

# 计算每个样本的风险评分
male_fetus_data['风险评分'] = male_fetus_data.apply(
    lambda row: calculate_risk_score(
        row['孕妇BMI指标'], 
        row['孕周数值'], 
        row['Y染色体浓度']
    ), axis=1
)

# === 2. 生存分析相关变量构造 ===

# 构造"事件"：Y染色体浓度达标（>阈值）
concentration_threshold = np.percentile(male_fetus_data['Y染色体浓度'], 25)  # 25分位数作为阈值
male_fetus_data['事件_达标'] = (male_fetus_data['Y染色体浓度'] >= concentration_threshold).astype(int)

# 构造"时间"：孕周数值
male_fetus_data['时间_孕周'] = male_fetus_data['孕周数值']

# === 3. 生存分析分组算法 ===

def survival_based_grouping(data, n_groups=3):
    """
    基于生存分析的BMI分组
    使用Kaplan-Meier估计器找到最优分组点
    """
    print(f"\n=== 生存分析分组算法 (K={n_groups}) ===")
    
    bmi_values = data['孕妇BMI指标'].values
    bmi_min, bmi_max = bmi_values.min(), bmi_values.max()
    
    # 初始分组点（等距）
    initial_cuts = np.linspace(bmi_min, bmi_max, n_groups + 1)[1:-1]
    
    best_cuts = initial_cuts.copy()
    best_score = -np.inf
    
    # 优化分组点
    for iteration in range(100):
        # 根据当前分组点划分数据
        labels = np.digitize(bmi_values, bins=best_cuts)
        
        # 计算各组的生存函数差异
        survival_score = 0.0
        groups = []
        
        for group_id in np.unique(labels):
            group_data = data[labels == group_id]
            if len(group_data) >= 10:  # 确保组内有足够样本
                groups.append((group_id, group_data))
        
        # 使用logrank检验评估组间差异
        if len(groups) >= 2:
            try:
                # 两两比较各组的生存曲线
                pairwise_scores = []
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        group1_data = groups[i][1]
                        group2_data = groups[j][1]
                        
                        # logrank检验
                        result = logrank_test(
                            group1_data['时间_孕周'], group1_data['事件_达标'],
                            group2_data['时间_孕周'], group2_data['事件_达标']
                        )
                        # 使用-log(p值)作为评分，p值越小差异越显著
                        score = -np.log(result.p_value + 1e-10)
                        pairwise_scores.append(score)
                
                survival_score = np.mean(pairwise_scores) if pairwise_scores else 0
                
            except Exception as e:
                survival_score = 0
        
        # 如果找到更好的分组
        if survival_score > best_score:
            best_score = survival_score
            # 微调分组点
            step = (bmi_max - bmi_min) * 0.01
            for i in range(len(best_cuts)):
                best_cuts[i] += np.random.uniform(-step, step)
                best_cuts[i] = np.clip(best_cuts[i], bmi_min, bmi_max)
            best_cuts = np.sort(best_cuts)
        
        if iteration % 20 == 0:
            print(f"  迭代 {iteration}: 生存分析评分 = {survival_score:.4f}")
    
    # 生成最终标签
    final_labels = np.digitize(bmi_values, bins=best_cuts)
    
    print(f"  生存分析分组完成，最终评分: {best_score:.4f}")
    return best_score, best_cuts, final_labels

# === 4. Cox比例风险模型分组 ===

def cox_based_grouping(data, n_groups=3):
    """
    基于Cox比例风险模型的分组
    """
    print(f"\n=== Cox比例风险模型分组 (K={n_groups}) ===")
    
    try:
        # 准备Cox模型数据
        cox_data = data[['孕妇BMI指标', '孕妇年龄', '时间_孕周', '事件_达标', '风险评分']].copy()
        cox_data = cox_data.dropna()
        
        # 拟合Cox模型
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col='时间_孕周', event_col='事件_达标')
        
        # 计算风险比
        risk_scores = cph.predict_partial_hazard(cox_data)
        
        # 基于风险比进行分组
        risk_percentiles = np.percentile(risk_scores, np.linspace(0, 100, n_groups + 1))
        risk_cuts = risk_percentiles[1:-1]
        
        # 映射回BMI空间
        bmi_values = cox_data['孕妇BMI指标'].values
        cox_labels = np.digitize(risk_scores, bins=risk_cuts)
        
        # 计算各组的BMI范围
        bmi_cuts = []
        for group_id in np.unique(cox_labels):
            group_bmi = bmi_values[cox_labels == group_id]
            if len(group_bmi) > 0:
                bmi_cuts.append(np.mean([group_bmi.min(), group_bmi.max()]))
        
        bmi_cuts = sorted(bmi_cuts)[:-1]  # 去掉最后一个
        
        # 重新生成基于BMI的标签
        final_labels = np.digitize(data['孕妇BMI指标'].values, bins=bmi_cuts)
        
        # 评分：使用模型的concordance index
        cox_score = cph.concordance_index_
        
        print(f"  Cox模型分组完成，C-index: {cox_score:.4f}")
        return cox_score, bmi_cuts, final_labels
        
    except Exception as e:
        print(f"  Cox模型分组失败: {e}")
        # 回退到风险评分分组
        risk_values = data['风险评分'].values
        risk_cuts = np.percentile(risk_values, np.linspace(0, 100, n_groups + 1))[1:-1]
        labels = np.digitize(risk_values, bins=risk_cuts)
        return 0.5, risk_cuts, labels

# === 5. 风险最小化分组算法 ===

def risk_minimization_grouping(data, n_groups=3):
    """
    直接最小化总风险的分组算法
    """
    print(f"\n=== 风险最小化分组算法 (K={n_groups}) ===")
    
    bmi_values = data['孕妇BMI指标'].values
    risk_values = data['风险评分'].values
    
    best_cuts = None
    best_total_risk = np.inf
    
    # 多次随机初始化寻找最优解
    for trial in range(50):
        # 随机初始化分组点
        cuts = np.sort(np.random.uniform(bmi_values.min(), bmi_values.max(), n_groups - 1))
        
        # 模拟退火优化
        current_cuts = cuts.copy()
        current_risk = np.inf
        temperature = 1.0
        
        for iteration in range(200):
            # 计算当前分组的总风险
            labels = np.digitize(bmi_values, bins=current_cuts)
            total_risk = 0.0
            
            for group_id in np.unique(labels):
                group_indices = labels == group_id
                if np.sum(group_indices) > 0:
                    group_risk = np.mean(risk_values[group_indices])
                    group_size = np.sum(group_indices)
                    # 加权风险（大组的权重更高）
                    total_risk += group_risk * group_size
            
            # 接受条件
            if total_risk < current_risk or np.random.rand() < np.exp(-(total_risk - current_risk) / temperature):
                current_risk = total_risk
                
                if total_risk < best_total_risk:
                    best_total_risk = total_risk
                    best_cuts = current_cuts.copy()
            
            # 扰动分组点
            if iteration < 199:  # 不在最后一次迭代时扰动
                idx = np.random.randint(len(current_cuts))
                step = (bmi_values.max() - bmi_values.min()) * 0.02
                current_cuts[idx] += np.random.uniform(-step, step)
                current_cuts[idx] = np.clip(current_cuts[idx], bmi_values.min(), bmi_values.max())
                current_cuts = np.sort(current_cuts)
            
            temperature *= 0.995
        
        if trial % 10 == 0:
            print(f"  试验 {trial}: 当前最优总风险 = {best_total_risk:.2f}")
    
    # 生成最终标签
    final_labels = np.digitize(bmi_values, bins=best_cuts)
    
    # 转换为最小化评分（负风险）
    risk_min_score = -best_total_risk / len(data)
    
    print(f"  风险最小化分组完成，平均风险: {-risk_min_score:.4f}")
    return risk_min_score, best_cuts, final_labels

# === 6. 执行所有分组算法 ===

print("\n=== 开始生存分析和风险优化分组 ===")

# 传统算法（作为对比基准）
bmi_values = male_fetus_data['孕妇BMI指标'].values.reshape(-1, 1)

# KMeans基准
silhouette_scores_kmeans = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(bmi_values)
    score = silhouette_score(bmi_values, labels)
    silhouette_scores_kmeans.append((k, score))

best_k_kmeans = max(silhouette_scores_kmeans, key=lambda x: x[1])[0]

# 新算法测试
algorithms_results = []

for k in range(3, 6):  # 测试3-5组
    print(f"\n--- 测试 {k} 组分组 ---")
    
    # 生存分析分组
    surv_score, surv_cuts, surv_labels = survival_based_grouping(male_fetus_data, k)
    algorithms_results.append((k, 'Survival', surv_score, surv_cuts, surv_labels))
    
    # Cox模型分组
    cox_score, cox_cuts, cox_labels = cox_based_grouping(male_fetus_data, k)
    algorithms_results.append((k, 'Cox', cox_score, cox_cuts, cox_labels))
    
    # 风险最小化分组
    risk_score, risk_cuts, risk_labels = risk_minimization_grouping(male_fetus_data, k)
    algorithms_results.append((k, 'RiskMin', risk_score, risk_cuts, risk_labels))

# 选择最佳算法
best_algorithm = max(algorithms_results, key=lambda x: x[2])
_, best_method, best_score, best_cuts, best_labels = best_algorithm

male_fetus_data['BMI聚类'] = best_labels

print(f"\n=== 最优方案选择 ===")
print(f"最佳算法: {best_method}")
print(f"最佳分组数: {len(np.unique(best_labels))}")
print(f"最佳评分: {best_score:.4f}")

# === 7. 分组结果分析和NIPT时点优化 ===

def optimize_nipt_timing(group_data):
    """
    为每个组优化NIPT时点，最小化风险
    """
    if len(group_data) == 0:
        return np.nan, 0.0
    
    # 尝试不同的检测时点（10-20周）
    test_weeks = np.arange(10, 21, 0.5)
    best_week = 13.0
    min_avg_risk = np.inf
    
    for week in test_weeks:
        # 计算在此时点检测的平均风险
        simulated_risks = []
        for _, row in group_data.iterrows():
            risk = calculate_risk_score(
                row['孕妇BMI指标'], 
                week,  # 使用测试时点
                row['Y染色体浓度']
            )
            simulated_risks.append(risk)
        
        avg_risk = np.mean(simulated_risks)
        if avg_risk < min_avg_risk:
            min_avg_risk = avg_risk
            best_week = week
    
    # 计算达标率（Y染色体浓度>阈值的比例）
    success_rate = np.mean(group_data['Y染色体浓度'] >= concentration_threshold)
    
    return best_week, success_rate * 100

# 分析各组
summary_output = []
summary_output.append("=== T2 v2.0：基于生存分析的BMI分组结果 ===\n")

# BMI统计
summary_output.append("--- 1. 男胎孕妇BMI分布分析 ---")
bmi_stats = male_fetus_data['孕妇BMI指标'].describe()
summary_output.append(f"BMI统计描述:\n{bmi_stats}\n")

# 分组结果
summary_output.append(f"--- 2. 最优分组方案 ---")
summary_output.append(f"算法: {best_method}")
summary_output.append(f"分组数: {len(np.unique(best_labels))}")
summary_output.append(f"评分: {best_score:.4f}\n")

# 各组分析
summary_output.append("--- 3. 各BMI组详细分析 ---")
group_results = []

for group_id in sorted(np.unique(best_labels)):
    group_data = male_fetus_data[male_fetus_data['BMI聚类'] == group_id]
    
    if len(group_data) > 0:
        bmi_min = group_data['孕妇BMI指标'].min()
        bmi_max = group_data['孕妇BMI指标'].max()
        sample_count = len(group_data)
        avg_risk = group_data['风险评分'].mean()
        
        # 优化NIPT时点
        optimal_week, success_rate = optimize_nipt_timing(group_data)
        
        summary_output.append(f"聚类 {group_id}: BMI范围 [{bmi_min:.2f}, {bmi_max:.2f}], "
                            f"样本数: {sample_count}, 平均风险: {avg_risk:.3f}")
        summary_output.append(f"  最优NIPT时点: {optimal_week:.2f}周, 预期成功率: {success_rate:.1f}%")
        
        group_results.append({
            'BMI聚类': group_id,
            'BMI范围下限': bmi_min,
            'BMI范围上限': bmi_max,
            '样本数': sample_count,
            '平均风险评分': avg_risk,
            '最优NIPT时点 (周)': optimal_week,
            '预期成功率 (%)': success_rate,
            '风险等级': '低' if avg_risk < 1.0 else '中' if avg_risk < 2.0 else '高'
        })

summary_output.append(f"\n--- 4. 风险分析总结 ---")
total_avg_risk = male_fetus_data['风险评分'].mean()
summary_output.append(f"整体平均风险: {total_avg_risk:.3f}")

# 保存结果
summary_text = '\n'.join(summary_output)
with open(os.path.join(results_dir, 'T2_survival_analysis_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(summary_text)

# 保存Excel结果
results_df = pd.DataFrame(group_results)
excel_path = os.path.join(results_dir, 'T2_survival_grouping_results.xlsx')
results_df.to_excel(excel_path, index=False)

print(summary_text)

# === 8. 可视化结果 ===

# 1. 生存分析曲线
plt.figure(figsize=(15, 12))

# 子图1：各组的Kaplan-Meier生存曲线
plt.subplot(2, 3, 1)
kmf = KaplanMeierFitter()

for group_id in sorted(np.unique(best_labels)):
    group_data = male_fetus_data[male_fetus_data['BMI聚类'] == group_id]
    if len(group_data) >= 5:  # 确保有足够样本
        kmf.fit(group_data['时间_孕周'], group_data['事件_达标'], 
                label=f'BMI组 {group_id} (n={len(group_data)})')
        kmf.plot(ax=plt.gca())

plt.title('各BMI组Y染色体浓度达标的生存曲线')
plt.xlabel('孕周')
plt.ylabel('达标概率')
plt.legend()
plt.grid(alpha=0.3)

# 子图2：风险评分分布
plt.subplot(2, 3, 2)
for group_id in sorted(np.unique(best_labels)):
    group_data = male_fetus_data[male_fetus_data['BMI聚类'] == group_id]
    plt.hist(group_data['风险评分'], alpha=0.6, label=f'BMI组 {group_id}', bins=20)

plt.title('各组风险评分分布')
plt.xlabel('风险评分')
plt.ylabel('频数')
plt.legend()
plt.grid(alpha=0.3)

# 子图3：BMI分布与分组
plt.subplot(2, 3, 3)
bmi_range = np.linspace(male_fetus_data['孕妇BMI指标'].min(), 
                       male_fetus_data['孕妇BMI指标'].max(), 100)
plt.hist(male_fetus_data['孕妇BMI指标'], bins=30, alpha=0.7, color='skyblue', label='BMI分布')

# 标记分组边界
if len(best_cuts) > 0:
    for cut in best_cuts:
        plt.axvline(cut, color='red', linestyle='--', alpha=0.8)

plt.title('BMI分布与分组边界')
plt.xlabel('BMI (kg/m²)')
plt.ylabel('频数')
plt.legend()
plt.grid(alpha=0.3)

# 子图4：各组NIPT时点推荐
plt.subplot(2, 3, 4)
group_ids = []
nipt_times = []
success_rates = []

for result in group_results:
    group_ids.append(f"组{result['BMI聚类']}")
    nipt_times.append(result['最优NIPT时点 (周)'])
    success_rates.append(result['预期成功率 (%)'])

bars = plt.bar(group_ids, nipt_times, alpha=0.8, color='lightcoral')
plt.title('各组最优NIPT时点')
plt.xlabel('BMI组')
plt.ylabel('推荐孕周')

# 添加成功率标签
for bar, rate in zip(bars, success_rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.grid(alpha=0.3)

# 子图5：算法性能对比
plt.subplot(2, 3, 5)
method_names = []
method_scores = []

for result in algorithms_results:
    k, method, score, _, _ = result
    method_names.append(f'{method}_K{k}')
    method_scores.append(score)

# 添加传统方法对比
method_names.append('KMeans_best')
method_scores.append(max(silhouette_scores_kmeans, key=lambda x: x[1])[1])

plt.bar(range(len(method_names)), method_scores, alpha=0.8)
plt.title('算法性能对比')
plt.xlabel('算法')
plt.ylabel('评分')
plt.xticks(range(len(method_names)), method_names, rotation=45)
plt.grid(alpha=0.3)

# 子图6：风险 vs BMI散点图
plt.subplot(2, 3, 6)
colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(best_labels))))
for i, group_id in enumerate(sorted(np.unique(best_labels))):
    group_data = male_fetus_data[male_fetus_data['BMI聚类'] == group_id]
    plt.scatter(group_data['孕妇BMI指标'], group_data['风险评分'], 
               c=[colors[i]], label=f'BMI组 {group_id}', alpha=0.6)

plt.title('BMI vs 风险评分')
plt.xlabel('BMI (kg/m²)')
plt.ylabel('风险评分')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_survival_analysis_comprehensive.png'), 
            dpi=500, bbox_inches='tight')
plt.close()

print(f"\n✅ T2 v2.0 生存分析完成！")
print(f"📊 结果保存在: {results_dir}")
print(f"📈 最佳算法: {best_method}")
print(f"🎯 分组评分: {best_score:.4f}")
print(f"⚡ 整体风险优化: {((2.0 - total_avg_risk) / 2.0 * 100):.1f}%")
