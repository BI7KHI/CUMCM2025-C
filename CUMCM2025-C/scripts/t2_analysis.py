import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 创建结果目录
results_dir = os.path.join(project_root, 'results')
os.makedirs(results_dir, exist_ok=True)

# 读取数据（不使用列名）
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

# 数据预处理：只保留男胎数据（即Y染色体浓度非空的行）
male_fetus_data = data[data['Y染色体浓度'].notna()].copy()

# 转换数值型列的数据类型
def safe_float_convert(x):
    try:
        return float(x)
    except:
        return np.nan

# 需要转换为数值型的列
numeric_columns = ['孕妇年龄', '孕妇身高', '孕妇体重', '孕妇BMI指标',
                   '原始测序数据的总读段数', '总读段数中在参考基因组上比对的比例', '总读段数中重复读段的比例',
                   '总读段数中唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值',
                   '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度',
                   'X染色体浓度', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
                   '被过滤掉的读段数占总读段数的比例']

for col in numeric_columns:
    male_fetus_data[col] = male_fetus_data[col].apply(safe_float_convert)

# 将孕周转换为数值，如11w+6转换为11+6/7=11.857
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

male_fetus_data['孕周数值'] = male_fetus_data['孕妇本次检测时的孕周'].apply(convert_gestational_age)

# 过滤掉包含NaN值的行
male_fetus_data = male_fetus_data.dropna(subset=['孕妇BMI指标', 'Y染色体浓度', '孕周数值'])

# 任务T2：对男胎孕妇的BMI进行合理分组，确定每组的最佳NIPT时点

# 1. 分析BMI分布
plt.figure(figsize=(10, 6))
sns.histplot(male_fetus_data['孕妇BMI指标'], kde=True, bins=30)
plt.title('孕妇BMI分布')
plt.xlabel('BMI')
plt.ylabel('频数')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_bmi_distribution.png'), dpi=300)
plt.close()

# 2. 使用K-means聚类对BMI进行分组
# 确定最佳聚类数
bmi_values = male_fetus_data['孕妇BMI指标'].values.reshape(-1, 1)
silhouette_scores = []
for n_clusters in range(2, 8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(bmi_values)
    silhouette_avg = silhouette_score(bmi_values, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f'聚类数: {n_clusters}, 轮廓系数: {silhouette_avg}')

# 可视化轮廓系数
plt.figure(figsize=(10, 6))
plt.plot(range(2, 8), silhouette_scores, marker='o')
plt.title('不同聚类数的轮廓系数')
plt.xlabel('聚类数')
plt.ylabel('轮廓系数')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_silhouette_scores.png'), dpi=300)
plt.close()

# 选择最佳聚类数（这里我们选择4个聚类，与题目中提到的BMI分组类似）
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
male_fetus_data['BMI聚类'] = kmeans.fit_predict(bmi_values)

# 分析每个聚类的BMI范围
cluster_bmi_ranges = {}
for cluster in range(n_clusters):
    cluster_data = male_fetus_data[male_fetus_data['BMI聚类'] == cluster]
    min_bmi = cluster_data['孕妇BMI指标'].min()
    max_bmi = cluster_data['孕妇BMI指标'].max()
    cluster_bmi_ranges[cluster] = (min_bmi, max_bmi)
    print(f'聚类 {cluster}: BMI范围 [{min_bmi:.2f}, {max_bmi:.2f}]')

# 3. 对每个BMI聚类，确定最佳NIPT时点
# 首先，找出每个孕妇最早达到Y染色体浓度≥4%的孕周
pregnant_women = male_fetus_data['孕妇代码'].unique()
early达标数据 = []

for woman in pregnant_women:
    woman_data = male_fetus_data[male_fetus_data['孕妇代码'] == woman].sort_values('孕周数值')
    # 找到最早的Y染色体浓度≥4%的记录
    达标记录 = woman_data[woman_data['Y染色体浓度'] >= 0.04]  # 4%
    if len(达标记录) > 0:
        最早达标记录 = 达标记录.iloc[0]
        early达标数据.append({
            '孕妇代码': woman,
            'BMI': 最早达标记录['孕妇BMI指标'],
            'BMI聚类': 最早达标记录['BMI聚类'],
            '最早达标孕周': 最早达标记录['孕周数值'],
            'Y染色体浓度': 最早达标记录['Y染色体浓度']
        })
    else:
        # 如果没有达标记录，取最大孕周作为参考
        max_week_record = woman_data.iloc[-1]
        early达标数据.append({
            '孕妇代码': woman,
            'BMI': max_week_record['孕妇BMI指标'],
            'BMI聚类': max_week_record['BMI聚类'],
            '最早达标孕周': np.nan,  # 未达标
            'Y染色体浓度': max_week_record['Y染色体浓度']
        })

early达标_df = pd.DataFrame(early达标数据)

# 分析每个BMI聚类的最佳NIPT时点
cluster_optimal_times = {}
for cluster in range(n_clusters):
    cluster_data = early达标_df[early达标_df['BMI聚类'] == cluster]
    # 计算该聚类中已达标的孕妇比例
    达标比例 = cluster_data['最早达标孕周'].notna().mean()
    
    # 对于已达标的孕妇，计算最佳NIPT时点（这里选择达标孕周的平均值减去一个标准差，以确保较高的达标概率）
    达标数据 = cluster_data[cluster_data['最早达标孕周'].notna()]
    if len(达标数据) > 0:
        平均达标孕周 = 达标数据['最早达标孕周'].mean()
        标准差 = 达标数据['最早达标孕周'].std()
        # 为了安全起见，选择平均达标孕周作为最佳时点（或者平均达标孕周+0.5标准差，确保更高的达标概率）
        最佳时点 = 平均达标孕周 + 0.5 * 标准差
        cluster_optimal_times[cluster] = {
            'BMI范围': cluster_bmi_ranges[cluster],
            '达标比例': 达标比例,
            '平均达标孕周': 平均达标孕周,
            '标准差': 标准差,
            '最佳NIPT时点': 最佳时点
        }
        print(f'聚类 {cluster} (BMI {cluster_bmi_ranges[cluster][0]:.2f}-{cluster_bmi_ranges[cluster][1]:.2f}):')
        print(f'  达标比例: {达标比例:.2%}')
        print(f'  平均达标孕周: {平均达标孕周:.2f}周')
        print(f'  最佳NIPT时点: {最佳时点:.2f}周')
    else:
        cluster_optimal_times[cluster] = {
            'BMI范围': cluster_bmi_ranges[cluster],
            '达标比例': 0,
            '平均达标孕周': np.nan,
            '标准差': np.nan,
            '最佳NIPT时点': np.nan
        }
        print(f'聚类 {cluster} (BMI {cluster_bmi_ranges[cluster][0]:.2f}-{cluster_bmi_ranges[cluster][1]:.2f}):')
        print(f'  所有孕妇均未达标')

# 4. 分析不同BMI分组的风险
# 风险定义：如果检测时间过早，可能导致未达标，需要重新检测，增加成本和焦虑
# 同时，如果检测时间过晚，可能错过最佳治疗窗口期

# 计算每个BMI聚类的潜在风险评分
# 风险评分 = (最佳时点 - 12) * 0.1 + (1 - 达标比例) * 0.9
# 这里假设12周是理想的最早检测时间，权重可以根据实际情况调整
cluster_risks = {}
for cluster in cluster_optimal_times:
    if not np.isnan(cluster_optimal_times[cluster]['最佳NIPT时点']):
        时点风险 = max(0, cluster_optimal_times[cluster]['最佳NIPT时点'] - 12) * 0.1
        达标风险 = (1 - cluster_optimal_times[cluster]['达标比例']) * 0.9
        总风险 = 时点风险 + 达标风险
        cluster_risks[cluster] = {
            '时点风险': 时点风险,
            '达标风险': 达标风险,
            '总风险': 总风险
        }
        print(f'聚类 {cluster} 风险评分: 时点风险={时点风险:.2f}, 达标风险={达标风险:.2f}, 总风险={总风险:.2f}')

# 5. 分析检测误差对结果的影响
# 模拟检测误差：假设Y染色体浓度测量存在±5%的误差
male_fetus_data_with_error = male_fetus_data.copy()
# 添加随机误差
np.random.seed(42)
error_percentage = 0.05  # 5%误差
male_fetus_data_with_error['Y染色体浓度_误差'] = male_fetus_data_with_error['Y染色体浓度'] * \
    (1 + np.random.normal(0, error_percentage, len(male_fetus_data_with_error)))

# 重新计算考虑误差后的达标时间
pregnant_women = male_fetus_data_with_error['孕妇代码'].unique()
early达标数据_error = []

for woman in pregnant_women:
    woman_data = male_fetus_data_with_error[male_fetus_data_with_error['孕妇代码'] == woman].sort_values('孕周数值')
    # 找到最早的Y染色体浓度≥4%的记录（考虑误差）
    达标记录 = woman_data[woman_data['Y染色体浓度_误差'] >= 0.04]  # 4%
    if len(达标记录) > 0:
        最早达标记录 = 达标记录.iloc[0]
        early达标数据_error.append({
            '孕妇代码': woman,
            'BMI': 最早达标记录['孕妇BMI指标'],
            'BMI聚类': 最早达标记录['BMI聚类'],
            '最早达标孕周': 最早达标记录['孕周数值'],
            'Y染色体浓度_误差': 最早达标记录['Y染色体浓度_误差']
        })
    else:
        # 如果没有达标记录，取最大孕周作为参考
        max_week_record = woman_data.iloc[-1]
        early达标数据_error.append({
            '孕妇代码': woman,
            'BMI': max_week_record['孕妇BMI指标'],
            'BMI聚类': max_week_record['BMI聚类'],
            '最早达标孕周': np.nan,  # 未达标
            'Y染色体浓度_误差': max_week_record['Y染色体浓度_误差']
        })

early达标_df_error = pd.DataFrame(early达标数据_error)

# 分析考虑误差后每个BMI聚类的达标情况变化
print("\n考虑检测误差后的结果变化：")
for cluster in range(n_clusters):
    original_data = early达标_df[early达标_df['BMI聚类'] == cluster]
    error_data = early达标_df_error[early达标_df_error['BMI聚类'] == cluster]
    
    original_success_rate = original_data['最早达标孕周'].notna().mean()
    error_success_rate = error_data['最早达标孕周'].notna().mean()
    
    print(f'聚类 {cluster} (BMI {cluster_bmi_ranges[cluster][0]:.2f}-{cluster_bmi_ranges[cluster][1]:.2f}):')
    print(f'  原始达标比例: {original_success_rate:.2%}')
    print(f'  考虑误差后达标比例: {error_success_rate:.2%}')
    print(f'  变化: {(error_success_rate - original_success_rate):.2%}')

# 6. 可视化结果
# 6.1 BMI聚类结果
plt.figure(figsize=(12, 8))
sns.scatterplot(x='孕妇BMI指标', y='孕周数值', hue='BMI聚类', data=male_fetus_data, palette='viridis')
# 添加聚类中心垂直线
for cluster in range(n_clusters):
    cluster_center = kmeans.cluster_centers_[cluster][0]
    plt.axvline(x=cluster_center, color='r', linestyle='--', alpha=0.5)
plt.title('BMI聚类结果')
plt.xlabel('BMI')
plt.ylabel('孕周')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_bmi_clustering.png'), dpi=300)
plt.close()

# 6.2 各聚类的最佳NIPT时点比较
plt.figure(figsize=(12, 6))
clusters = list(cluster_optimal_times.keys())
optimal_times = [cluster_optimal_times[cluster]['最佳NIPT时点'] for cluster in clusters]
bmi_ranges = [f"{cluster_bmi_ranges[cluster][0]:.1f}-{cluster_bmi_ranges[cluster][1]:.1f}" for cluster in clusters]
success_rates = [cluster_optimal_times[cluster]['达标比例'] * 100 for cluster in clusters]

# 创建柱状图
bar_width = 0.35
x = np.arange(len(clusters))
fig, ax1 = plt.subplots(figsize=(12, 6))

bar1 = ax1.bar(x - bar_width/2, optimal_times, bar_width, label='最佳NIPT时点(周)')
ax1.set_xlabel('BMI聚类')
ax1.set_ylabel('最佳NIPT时点(周)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(x)
ax1.set_xticklabels(bmi_ranges)

# 创建第二个y轴显示达标比例
ax2 = ax1.twinx()
bar2 = ax2.bar(x + bar_width/2, success_rates, bar_width, label='达标比例(%)', color='green')
ax2.set_ylabel('达标比例(%)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('各BMI聚类的最佳NIPT时点和达标比例')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_optimal_nipt_times.png'), dpi=300)
plt.close()

# 保存结果到Excel
results_data = []
for cluster in cluster_optimal_times:
    cluster_data = cluster_optimal_times[cluster]
    results_data.append({
        'BMI聚类': cluster,
        'BMI范围下限': cluster_data['BMI范围'][0],
        'BMI范围上限': cluster_data['BMI范围'][1],
        '达标比例': cluster_data['达标比例'],
        '平均达标孕周': cluster_data['平均达标孕周'],
        '最佳NIPT时点': cluster_data['最佳NIPT时点']
    })

results_df = pd.DataFrame(results_data)
results_df.to_excel(os.path.join(results_dir, 'T2_bmi_grouping_results.xlsx'), index=False)

print("\nT2任务分析完成！结果已保存到results目录。")