import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import matplotlib.font_manager as fm

# 统一中文字体设置（优先使用项目根目录 fonts/ 与常用CJK字体）
def configure_chinese_font():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        fonts_dir = os.path.join(project_root, "fonts")
        if os.path.isdir(fonts_dir):
            for file_name in os.listdir(fonts_dir):
                if file_name.lower().endswith((".ttf", ".otf")):
                    try:
                        fm.fontManager.addfont(os.path.join(fonts_dir, file_name))
                    except Exception:
                        pass

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

        installed_families = set(f.name for f in fm.fontManager.ttflist)
        for family in candidate_families:
            if family in installed_families:
                plt.rcParams['font.family'] = [family]
                plt.rcParams['font.sans-serif'] = [family]
                return family
    except Exception:
        pass
    return None

chosen_font = configure_chinese_font()
if not chosen_font:
    print("Warning: 未检测到可用中文字体，建议运行项目根目录的 setup_chinese_fonts.py 后重试。")
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

# 初始化一个字符串构建器来存储摘要
summary_output = []

# 任务T2：对男胎孕妇的BMI进行合理分组，确定每组的最佳NIPT时点

# 1. 分析BMI分布
summary_output.append("--- 1. 孕妇BMI分布分析 ---")
bmi_description = male_fetus_data['孕妇BMI指标'].describe()
summary_output.append(f"BMI统计描述:\\n{bmi_description}\\n")

plt.figure(figsize=(10, 6))
sns.histplot(male_fetus_data['孕妇BMI指标'], kde=True, bins=30)
plt.title('孕妇BMI分布')
plt.xlabel('BMI')
plt.ylabel('频数')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_bmi_distribution.png'), dpi=300)
plt.close()
summary_output.append("BMI分布直方图已保存为 'T2_bmi_distribution.png'。\\n")

# 2. 使用K-means聚类对BMI进行分组
summary_output.append("--- 2. 使用K-means聚类对BMI进行分组 ---")
# 确定最佳聚类数
bmi_values = male_fetus_data['孕妇BMI指标'].values.reshape(-1, 1)
silhouette_scores = []
summary_output.append("轮廓系数分析 (用于确定最佳聚类数):")
for n_clusters in range(2, 8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(bmi_values)
    silhouette_avg = silhouette_score(bmi_values, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    summary_output.append(f'  - 聚类数: {n_clusters}, 轮廓系数: {silhouette_avg:.4f}')

# 可视化轮廓系数
plt.figure(figsize=(10, 6))
plt.plot(range(2, 8), silhouette_scores, marker='o')
plt.title('不同聚类数的轮廓系数')
plt.xlabel('聚类数')
plt.ylabel('轮廓系数')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_silhouette_scores.png'), dpi=300)
plt.close()
summary_output.append("轮廓系数可视化图已保存为 'T2_silhouette_scores.png'。")

# 根据轮廓系数和业务理解（例如，低、中、高、超高BMI），选择4作为最佳聚类数。
n_clusters = 4
summary_output.append(f"\\n选择的最佳聚类数为: {n_clusters}。这个选择是基于轮廓系数的分析以及与常规BMI分组（如偏瘦、正常、超重、肥胖）的对应关系。\\n")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
male_fetus_data['BMI聚类'] = kmeans.fit_predict(bmi_values)

# 分析每个聚类的BMI范围
summary_output.append("各BMI聚类的范围:")
cluster_bmi_ranges = {}
for cluster in sorted(male_fetus_data['BMI聚类'].unique()):
    cluster_data = male_fetus_data[male_fetus_data['BMI聚类'] == cluster]
    min_bmi = cluster_data['孕妇BMI指标'].min()
    max_bmi = cluster_data['孕妇BMI指标'].max()
    cluster_bmi_ranges[cluster] = (min_bmi, max_bmi)
    summary_output.append(f'  - 聚类 {cluster}: BMI范围 [{min_bmi:.2f}, {max_bmi:.2f}], 样本数: {len(cluster_data)}')
summary_output.append('')

# 3. 对每个BMI聚类，确定最佳NIPT时点
summary_output.append("--- 3. 各BMI聚类的最佳NIPT时点分析 ---")
# 定义“达标”：Y染色体浓度 >= 0.04 (4%)
Y_THRESHOLD = 0.04

# 找出每个孕妇最早达到Y染色体浓度阈值的孕周
pregnant_women = male_fetus_data['孕妇代码'].unique()
early_success_data = []

for woman in pregnant_women:
    woman_data = male_fetus_data[male_fetus_data['孕妇代码'] == woman].sort_values('孕周数值')
    # 找到最早的达标记录
    success_records = woman_data[woman_data['Y染色体浓度'] >= Y_THRESHOLD]
    if not success_records.empty:
        first_success_record = success_records.iloc[0]
        early_success_data.append({
            '孕妇代码': woman,
            'BMI': first_success_record['孕妇BMI指标'],
            'BMI聚类': first_success_record['BMI聚类'],
            '最早达标孕周': first_success_record['孕周数值'],
            'Y染色体浓度': first_success_record['Y染色体浓度']
        })
    else:
        # 如果没有达标记录，记录为未达标
        last_record = woman_data.iloc[-1]
        early_success_data.append({
            '孕妇代码': woman,
            'BMI': last_record['孕妇BMI指标'],
            'BMI聚类': last_record['BMI聚类'],
            '最早达标孕周': np.nan,  # 未达标
            'Y染色体浓度': last_record['Y染色体浓度']
        })

early_success_df = pd.DataFrame(early_success_data)

# 分析每个BMI聚类的最佳NIPT时点
cluster_optimal_times = {}
for cluster in sorted(early_success_df['BMI聚类'].unique()):
    cluster_data = early_success_df[early_success_df['BMI聚类'] == cluster]
    
    # 计算达标比例
    success_rate = cluster_data['最早达标孕周'].notna().mean()
    
    # 对于已达标的孕妇，计算统计数据
    successful_data = cluster_data.dropna(subset=['最早达标孕周'])
    if not successful_data.empty:
        mean_success_week = successful_data['最早达标孕周'].mean()
        std_success_week = successful_data['最早达标孕周'].std()
        # 最佳时点定义为平均达标孕周，以在覆盖大部分人群和避免过晚检测之间取得平衡。
        # 也可以考虑更保守的策略，如 mean + 0.5 * std，以提高首次检测成功率。
        # 这里我们选择平均值作为推荐时点。
        optimal_time = mean_success_week
        
        cluster_optimal_times[cluster] = {
            'BMI范围': cluster_bmi_ranges[cluster],
            '达标比例': success_rate,
            '平均达标孕周': mean_success_week,
            '达标孕周标准差': std_success_week,
            '推荐NIPT时点 (周)': optimal_time
        }
        summary_output.append(f'聚类 {cluster} (BMI {cluster_bmi_ranges[cluster][0]:.2f}-{cluster_bmi_ranges[cluster][1]:.2f}):')
        summary_output.append(f'  - 达标比例: {success_rate:.2%}')
        summary_output.append(f'  - 平均达标孕周: {mean_success_week:.2f}周')
        summary_output.append(f'  - 推荐NIPT时点: {optimal_time:.2f}周 (建议取整为 {np.ceil(optimal_time):.0f} 周后)')
    else:
        cluster_optimal_times[cluster] = {
            'BMI范围': cluster_bmi_ranges[cluster],
            '达标比例': 0,
            '平均达标孕周': np.nan,
            '达标孕周标准差': np.nan,
            '推荐NIPT时点 (周)': np.nan
        }
        summary_output.append(f'聚类 {cluster} (BMI {cluster_bmi_ranges[cluster][0]:.2f}-{cluster_bmi_ranges[cluster][1]:.2f}):')
        summary_output.append(f'  - 所有孕妇均未达标 (Y染色体浓度 < {Y_THRESHOLD*100}%)')
summary_output.append('')

# 4. 分析不同BMI分组的风险
summary_output.append("--- 4. 不同BMI分组的检测风险分析 ---")
summary_output.append("风险定义: 综合考虑“过早检测导致失败”和“过晚检测错过干预窗口”的可能。")
summary_output.append("风险评分公式: (推荐NIPT时点 - 12) * 0.1 + (1 - 达标比例) * 0.9")
summary_output.append("  - 假设理想的最早检测时间为12周。")
summary_output.append("  - 权重分配上，更关注“检测失败”的风险。\\n")

cluster_risks = {}
for cluster in cluster_optimal_times:
    if not np.isnan(cluster_optimal_times[cluster]['推荐NIPT时点 (周)']):
        time_risk = max(0, cluster_optimal_times[cluster]['推荐NIPT时点 (周)'] - 12) * 0.1
        success_risk = (1 - cluster_optimal_times[cluster]['达标比例']) * 0.9
        total_risk = time_risk + success_risk
        cluster_risks[cluster] = {
            '时点风险': time_risk,
            '达标风险': success_risk,
            '总风险': total_risk
        }
        summary_output.append(f'聚类 {cluster} 风险评分:')
        summary_output.append(f'  - 时点风险: {time_risk:.3f}')
        summary_output.append(f'  - 达标风险: {success_risk:.3f}')
        summary_output.append(f'  - 总风险: {total_risk:.3f}')
summary_output.append('')

# 5. 分析检测误差对结果的影响
summary_output.append("--- 5. 检测误差对结果影响的敏感性分析 ---")
# 模拟检测误差：假设Y染色体浓度测量存在±5%的相对误差
error_percentage = 0.05
summary_output.append(f"模拟 {error_percentage:.0%} 的随机测量误差...\\n")

np.random.seed(42)
male_fetus_data_with_error = male_fetus_data.copy()
male_fetus_data_with_error['Y染色体浓度_误差'] = male_fetus_data_with_error['Y染色体浓度'] * \
    (1 + np.random.normal(0, error_percentage, len(male_fetus_data_with_error)))

# 重新计算考虑误差后的达标时间
early_success_data_error = []
for woman in pregnant_women:
    woman_data = male_fetus_data_with_error[male_fetus_data_with_error['孕妇代码'] == woman].sort_values('孕周数值')
    success_records = woman_data[woman_data['Y染色体浓度_误差'] >= Y_THRESHOLD]
    if not success_records.empty:
        first_success_record = success_records.iloc[0]
        early_success_data_error.append({
            '孕妇代码': woman,
            'BMI聚类': first_success_record['BMI聚类'],
            '最早达标孕周': first_success_record['孕周数值'],
        })
    else:
        last_record = woman_data.iloc[-1]
        early_success_data_error.append({
            '孕妇代码': woman,
            'BMI聚类': last_record['BMI聚类'],
            '最早达标孕周': np.nan,
        })

early_success_df_error = pd.DataFrame(early_success_data_error)

# 分析考虑误差后每个BMI聚类的达标情况变化
summary_output.append("考虑检测误差后的达标比例变化:")
for cluster in sorted(early_success_df['BMI聚类'].unique()):
    original_success_rate = early_success_df[early_success_df['BMI聚类'] == cluster]['最早达标孕周'].notna().mean()
    error_success_rate = early_success_df_error[early_success_df_error['BMI聚类'] == cluster]['最早达标孕周'].notna().mean()
    
    summary_output.append(f'聚类 {cluster} (BMI {cluster_bmi_ranges[cluster][0]:.2f}-{cluster_bmi_ranges[cluster][1]:.2f}):')
    summary_output.append(f'  - 原始达标比例: {original_success_rate:.2%}')
    summary_output.append(f'  - 考虑误差后达标比例: {error_success_rate:.2%}')
    summary_output.append(f'  - 变化: {(error_success_rate - original_success_rate):+.2%}')
summary_output.append('')

# 6. 可视化结果
summary_output.append("--- 6. 可视化结果 ---")
# 6.1 BMI聚类结果
plt.figure(figsize=(12, 8))
sns.scatterplot(x='孕妇BMI指标', y='孕周数值', hue='BMI聚类', data=male_fetus_data, palette='viridis', legend='full')
# 添加聚类中心和范围
for cluster in sorted(cluster_bmi_ranges.keys()):
    min_bmi, max_bmi = cluster_bmi_ranges[cluster]
    plt.axvline(x=min_bmi, color='grey', linestyle='--', alpha=0.5)
    plt.axvline(x=max_bmi, color='grey', linestyle='--', alpha=0.5)
plt.title('BMI聚类结果及孕周分布')
plt.xlabel('BMI')
plt.ylabel('孕周 (周)')
plt.legend(title='BMI聚类')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_bmi_clustering.png'), dpi=300)
plt.close()
summary_output.append("BMI聚类结果图已保存为 'T2_bmi_clustering.png'。")

# 6.2 各聚类的最佳NIPT时点比较
plt.figure(figsize=(12, 6))
clusters = list(cluster_optimal_times.keys())
optimal_times = [cluster_optimal_times[c]['推荐NIPT时点 (周)'] for c in clusters]
bmi_ranges_str = [f"聚类 {c}\\n({cluster_bmi_ranges[c][0]:.1f}-{cluster_bmi_ranges[c][1]:.1f})" for c in clusters]
success_rates_pct = [cluster_optimal_times[c]['达标比例'] * 100 for c in clusters]

fig, ax1 = plt.subplots(figsize=(12, 7))

# 柱状图表示推荐时点
bar1 = ax1.bar(bmi_ranges_str, optimal_times, label='推荐NIPT时点 (周)', color='skyblue')
ax1.set_xlabel('BMI聚类及范围')
ax1.set_ylabel('推荐NIPT时点 (周)', color='darkblue')
ax1.tick_params(axis='y', labelcolor='darkblue')
ax1.set_ylim(bottom=10)

# 第二个y轴显示达标比例
ax2 = ax1.twinx()
line1 = ax2.plot(bmi_ranges_str, success_rates_pct, 'o-', color='green', label='达标比例 (%)')
ax2.set_ylabel('达标比例 (%)', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(0, 105)

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center')

plt.title('各BMI聚类的推荐NIPT时点和达标比例')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_optimal_nipt_times.png'), dpi=300)
plt.close()
summary_output.append("各BMI组的推荐NIPT时点和达标比例图已保存为 'T2_optimal_nipt_times.png'。\\n")

# 7. 保存结果到Excel和总结文件
summary_output.append("--- 7. 结果汇总 ---")
results_data = []
for cluster in sorted(cluster_optimal_times.keys()):
    cluster_data = cluster_optimal_times[cluster]
    results_data.append({
        'BMI聚类': cluster,
        'BMI范围下限': cluster_data['BMI范围'][0],
        'BMI范围上限': cluster_data['BMI范围'][1],
        '达标比例': f"{cluster_data['达标比例']:.2%}",
        '平均达标孕周': f"{cluster_data['平均达标孕周']:.2f}",
        '推荐NIPT时点 (周)': f"{cluster_data['推荐NIPT时点 (周)']:.2f}",
        '总风险评分': f"{cluster_risks.get(cluster, {}).get('总风险', np.nan):.3f}"
    })

results_df = pd.DataFrame(results_data)
excel_path = os.path.join(results_dir, 'T2_bmi_grouping_results.xlsx')
results_df.to_excel(excel_path, index=False)
summary_output.append(f"详细结果已保存到Excel文件: '{excel_path}'")

# 生成最终总结
final_summary = "\\n".join(summary_output)
summary_file_path = os.path.join(results_dir, 'T2_analysis_summary.txt')
with open(summary_file_path, 'w', encoding='utf-8') as f:
    f.write("=============== T2任务：男胎孕妇BMI分组及NIPT时点分析报告 ===============\\n\\n")
    f.write(final_summary)
    f.write("\\n\\n================================= 报告结束 =================================")

print(f"\\nT2任务分析完成！详细分析报告已保存到: {summary_file_path}")