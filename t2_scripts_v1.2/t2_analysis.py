import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import os
import random
import matplotlib.font_manager as fm
import urllib.request

# 健壮的中文字体配置（优先本地 fonts/，尝试下载 Noto Sans CJK）
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
    # 兜底，至少保证负号正常显示
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("未找到中文字体，使用默认字体 DejaVu Sans")

plt.rcParams['axes.unicode_minus'] = False

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 创建结果目录（v1.2 专用）
results_dir = os.path.join(project_root, 'results_v1.2')
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

# 任务T2 v1.2：对男胎孕妇的BMI进行合理分组（KMeans vs GMM 自动选型）
summary_output.append("=== T2 v1.2：男胎孕妇BMI分组（KMeans vs GMM） ===\n")

# 1. 分析BMI分布
summary_output.append("--- 1. 孕妇BMI分布分析 ---")
bmi_description = male_fetus_data['孕妇BMI指标'].describe()
summary_output.append(f"BMI统计描述:\n{bmi_description}\n")

plt.figure(figsize=(10, 6))
sns.histplot(male_fetus_data['孕妇BMI指标'], kde=True, bins=30)
plt.title('孕妇BMI分布')
plt.xlabel('BMI')
plt.ylabel('频数')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_bmi_distribution.png'), dpi=500)
plt.close()
summary_output.append("BMI分布直方图已保存为 'T2_bmi_distribution.png'。\n")

# 2. 选择聚类数与模型（KMeans vs GMM）
summary_output.append("--- 2. 模型选择：KMeans vs GMM 与 进化/退火阈值优化 ---")
bmi_values = male_fetus_data['孕妇BMI指标'].values.reshape(-1, 1)

k_range = range(2, 8)
silhouette_scores_kmeans = []
silhouette_scores_gmm = []
bic_scores_gmm = []

for n_clusters in k_range:
    # KMeans 轮廓系数
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_k = kmeans.fit_predict(bmi_values)
    sil_k = silhouette_score(bmi_values, labels_k)
    silhouette_scores_kmeans.append((n_clusters, sil_k))

    # GMM BIC 与 轮廓
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    gmm.fit(bmi_values)
    labels_g = gmm.predict(bmi_values)
    bic_g = gmm.bic(bmi_values)
    sil_g = silhouette_score(bmi_values, labels_g)
    silhouette_scores_gmm.append((n_clusters, sil_g))
    bic_scores_gmm.append((n_clusters, bic_g))

# 选择最佳K（KMeans按轮廓最大；GMM按BIC最小）
best_k_kmeans, best_sil_kmeans = max(silhouette_scores_kmeans, key=lambda x: x[1])
best_k_gmm_bic, best_bic_gmm = min(bic_scores_gmm, key=lambda x: x[1])
best_k_gmm_sil, best_sil_gmm = max(silhouette_scores_gmm, key=lambda x: x[1])

# 简单策略：若GMM的BIC很低且轮廓不差于KMeans过多，则选GMM，否则选KMeans
use_gmm = (best_bic_gmm < np.median([b for _, b in bic_scores_gmm])) and (best_sil_gmm >= best_sil_kmeans - 0.03)

if use_gmm:
    selected_k = best_k_gmm_bic
    model_name = 'GMM'
    model = GaussianMixture(n_components=selected_k, covariance_type='full', random_state=42)
    model.fit(bmi_values)
    cluster_labels = model.predict(bmi_values)
else:
    selected_k = best_k_kmeans
    model_name = 'KMeans'
    model = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(bmi_values)

# === 进化/退火：连续阈值分割（将BMI按若干阈值划分为k组） ===
def evaluate_thresholds(bmi, cuts):
    # cuts 为升序阈值列表，将区间映射到标签
    labels = np.digitize(bmi, bins=np.array(cuts))
    # 至少需要两组
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -np.inf, labels
    
    # 检查每个标签是否有足够样本
    for label in unique_labels:
        if np.sum(labels == label) < 2:
            return -np.inf, labels
    
    # 对于大数据集，随机采样加速轮廓系数计算
    if len(bmi) > 800:
        sample_size = min(400, len(bmi))
        sample_idx = np.random.choice(len(bmi), sample_size, replace=False)
        sample_bmi = bmi[sample_idx]
        sample_labels = labels[sample_idx]
        try:
            sil = silhouette_score(sample_bmi.reshape(-1,1), sample_labels)
        except Exception:
            sil = -1
    else:
        try:
            sil = silhouette_score(bmi.reshape(-1,1), labels)
        except Exception:
            sil = -1
    
    # 类内方差总和（越小越好）
    var_sum = 0.0
    for g in unique_labels:
        vals = bmi[labels == g]
        if len(vals) > 1:
            var_sum += np.var(vals)
    # 综合评分：轮廓占主导，方差有轻微惩罚
    score = sil - 0.01 * var_sum
    return score, labels

# 模拟退火优化阈值（优化版）
def simulated_annealing(bmi, k, iters=500, init_temp=1.0, cooling=0.99):
    bmi_min, bmi_max = np.min(bmi), np.max(bmi)
    # 初始化均匀切分
    cuts = list(np.linspace(bmi_min, bmi_max, k+1)[1:-1]) if k > 1 else []
    best_cuts = cuts[:]
    best_score, _ = evaluate_thresholds(bmi, best_cuts)
    current_score = best_score
    temp = init_temp
    
    # 早停机制
    no_improve_count = 0
    max_no_improve = 100
    
    print(f"开始模拟退火优化 k={k} 组分组...")
    for i in range(iters):
        if k <= 1:
            break
        
        idx = random.randrange(0, k-1)
        new_cuts = cuts[:]
        # 邻域扰动
        step = (bmi_max - bmi_min) * 0.015  # 减小步长
        new_cuts[idx] = np.clip(new_cuts[idx] + np.random.uniform(-step, step), bmi_min, bmi_max)
        new_cuts = sorted(new_cuts)
        new_score, _ = evaluate_thresholds(bmi, new_cuts)
        
        # 接受条件
        if (new_score > current_score) or (temp > 1e-6 and np.exp((new_score - current_score) / temp) > np.random.rand()):
            cuts = new_cuts
            current_score = new_score
            if new_score > best_score:
                best_score, best_cuts = new_score, new_cuts
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # 早停
        if no_improve_count >= max_no_improve:
            print(f"  SA 早停于第 {i+1} 次迭代")
            break
            
        temp *= cooling
        
        # 每100次迭代输出进度
        if (i + 1) % 100 == 0:
            print(f"  SA 第 {i+1}/{iters} 次迭代, 当前最佳分数: {best_score:.4f}")
    
    print(f"  SA 完成，最终分数: {best_score:.4f}")
    return best_score, best_cuts

# 简易遗传算法优化阈值（优化版）
def genetic_optimize(bmi, k, pop_size=30, gens=50, mutation_rate=0.15, elite=0.2):
    bmi_min, bmi_max = np.min(bmi), np.max(bmi)
    if k <= 1:
        return evaluate_thresholds(bmi, [])[0], []
    
    def random_cuts():
        cuts = list(np.sort(np.random.uniform(bmi_min, bmi_max, size=k-1)))
        return cuts
    
    # 初始种群包含一些启发式解
    population = []
    # 添加均匀分割解
    uniform_cuts = list(np.linspace(bmi_min, bmi_max, k+1)[1:-1])
    population.append(uniform_cuts)
    # 添加随机解
    population.extend([random_cuts() for _ in range(pop_size-1)])
    
    def fitness(cuts):
        score, _ = evaluate_thresholds(bmi, cuts)
        return score if score != -np.inf else -1e6
    
    elite_count = max(1, int(pop_size * elite))
    best_overall_score = -np.inf
    no_improve_gens = 0
    max_no_improve = 15
    
    print(f"开始遗传算法优化 k={k} 组分组...")
    for gen in range(gens):
        scored = sorted(((fitness(c), c) for c in population), key=lambda x: x[0], reverse=True)
        
        # 记录最佳解
        current_best = scored[0][0]
        if current_best > best_overall_score:
            best_overall_score = current_best
            no_improve_gens = 0
        else:
            no_improve_gens += 1
        
        # 早停
        if no_improve_gens >= max_no_improve:
            print(f"  GA 早停于第 {gen+1} 代")
            break
        
        # 精英保留
        population = [c for _, c in scored[:elite_count]]
        
        # 交叉和变异产生新个体
        while len(population) < pop_size:
            p1 = random.choice(scored[:elite_count])[1]
            p2 = random.choice(scored[:elite_count])[1]
            cross = sorted([np.random.choice([a,b]) for a,b in zip(p1, p2)])
            
            # 变异
            if np.random.rand() < mutation_rate:
                idx = random.randrange(0, k-1)
                step = (bmi_max - bmi_min) * 0.02  # 减小步长
                cross[idx] = np.clip(cross[idx] + np.random.uniform(-step, step), bmi_min, bmi_max)
                cross = sorted(cross)
            population.append(cross)
        
        # 每10代输出进度
        if (gen + 1) % 10 == 0:
            print(f"  GA 第 {gen+1}/{gens} 代, 当前最佳分数: {current_best:.4f}")
    
    best_score, best_cuts = max(((fitness(c), c) for c in population), key=lambda x: x[0])
    print(f"  GA 完成，最终分数: {best_score:.4f}")
    return best_score, best_cuts

# 针对 k=3..5 分别尝试退火与遗传，取最佳（减少范围以提升速度）
evo_candidates = []
print("\n=== 开始进化算法优化 ===")
for k in range(3, 6):  # 减少到 3-5 组
    print(f"\n--- 测试 {k} 组分组 ---")
    sa_score, sa_cuts = simulated_annealing(bmi_values.flatten(), k)
    ga_score, ga_cuts = genetic_optimize(bmi_values.flatten(), k)
    evo_candidates.append((k, 'SA', sa_score, sa_cuts))
    evo_candidates.append((k, 'GA', ga_score, ga_cuts))
    print(f"k={k}: SA得分={sa_score:.4f}, GA得分={ga_score:.4f}")

best_evo = max(evo_candidates, key=lambda x: x[2])
_, evo_method, evo_score, evo_cuts = best_evo
_, evo_labels = evaluate_thresholds(bmi_values.flatten(), evo_cuts)

# 选择三类方案中最佳（KMeans / GMM / Evo）
# 使用统一评分：轮廓系数
labels_kmeans = KMeans(n_clusters=best_k_kmeans, random_state=42, n_init=10).fit_predict(bmi_values)
labels_gmm = GaussianMixture(n_components=best_k_gmm_bic, covariance_type='full', random_state=42).fit(bmi_values).predict(bmi_values)
score_kmeans = silhouette_score(bmi_values, labels_kmeans)
score_gmm = silhouette_score(bmi_values, labels_gmm)
score_evo = silhouette_score(bmi_values, evo_labels) if len(np.unique(evo_labels))>1 else -1

method_scores = [('KMeans', best_k_kmeans, score_kmeans, labels_kmeans),
                 ('GMM', best_k_gmm_bic, score_gmm, labels_gmm),
                 (f'Evo({evo_method})', len(np.unique(evo_labels)), score_evo, evo_labels)]
method_name, selected_k, best_score_val, best_labels = max(method_scores, key=lambda x: x[2])

male_fetus_data['BMI聚类'] = best_labels
summary_output.append(f"模型选择：{method_name} | 选定聚类数: {selected_k} | 轮廓系数: {best_score_val:.4f}\n")

# 可视化指标
plt.figure(figsize=(10, 6))
plt.plot([k for k, _ in silhouette_scores_kmeans], [s for _, s in silhouette_scores_kmeans], 'o-', label='KMeans Silhouette')
plt.plot([k for k, _ in silhouette_scores_gmm], [s for _, s in silhouette_scores_gmm], 'o-', label='GMM Silhouette')
plt.title('不同聚类数的轮廓系数对比')
plt.xlabel('聚类数'); plt.ylabel('轮廓系数')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_silhouette_compare.png'), dpi=500)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot([k for k, _ in bic_scores_gmm], [b for _, b in bic_scores_gmm], 'o-', color='tab:orange')
plt.title('GMM 不同聚类数的BIC')
plt.xlabel('聚类数'); plt.ylabel('BIC (越小越好)')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_gmm_bic.png'), dpi=500)
plt.close()

# 3. 分析每个聚类的BMI范围
summary_output.append("--- 3. 各BMI聚类范围 ---")
cluster_bmi_ranges = {}
for cluster in sorted(male_fetus_data['BMI聚类'].unique()):
    cluster_data = male_fetus_data[male_fetus_data['BMI聚类'] == cluster]
    min_bmi = cluster_data['孕妇BMI指标'].min()
    max_bmi = cluster_data['孕妇BMI指标'].max()
    cluster_bmi_ranges[cluster] = (min_bmi, max_bmi)
    summary_output.append(f'聚类 {cluster}: BMI范围 [{min_bmi:.2f}, {max_bmi:.2f}], 样本数: {len(cluster_data)}')
summary_output.append('')

# 4. 根据分组估计最佳NIPT时点（沿用原始逻辑）
summary_output.append("--- 4. 各BMI聚类的最佳NIPT时点分析 ---")
Y_THRESHOLD = 0.04
pregnant_women = male_fetus_data['孕妇代码'].unique()

early_success_data = []
for woman in pregnant_women:
    woman_data = male_fetus_data[male_fetus_data['孕妇代码'] == woman].sort_values('孕周数值')
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
        last_record = woman_data.iloc[-1]
        early_success_data.append({
            '孕妇代码': woman,
            'BMI': last_record['孕妇BMI指标'],
            'BMI聚类': last_record['BMI聚类'],
            '最早达标孕周': np.nan,
            'Y染色体浓度': last_record['Y染色体浓度']
        })

early_success_df = pd.DataFrame(early_success_data)

cluster_optimal_times = {}
for cluster in sorted(early_success_df['BMI聚类'].unique()):
    cluster_data = early_success_df[early_success_df['BMI聚类'] == cluster]
    success_rate = cluster_data['最早达标孕周'].notna().mean()
    successful_data = cluster_data.dropna(subset=['最早达标孕周'])
    if not successful_data.empty:
        mean_success_week = successful_data['最早达标孕周'].mean()
        std_success_week = successful_data['最早达标孕周'].std()
        optimal_time = mean_success_week
        cluster_optimal_times[cluster] = {
            'BMI范围': cluster_bmi_ranges[cluster],
            '达标比例': success_rate,
            '平均达标孕周': mean_success_week,
            '达标孕周标准差': std_success_week,
            '推荐NIPT时点 (周)': optimal_time
        }
    else:
        cluster_optimal_times[cluster] = {
            'BMI范围': cluster_bmi_ranges[cluster],
            '达标比例': 0,
            '平均达标孕周': np.nan,
            '达标孕周标准差': np.nan,
            '推荐NIPT时点 (周)': np.nan
        }

# 5. 可视化结果（更高DPI）
plt.figure(figsize=(12, 8))
sns.scatterplot(x='孕妇BMI指标', y='孕周数值', hue='BMI聚类', data=male_fetus_data, palette='viridis', legend='full')
for cluster in sorted(cluster_bmi_ranges.keys()):
    min_bmi, max_bmi = cluster_bmi_ranges[cluster]
    plt.axvline(x=min_bmi, color='grey', linestyle='--', alpha=0.5)
    plt.axvline(x=max_bmi, color='grey', linestyle='--', alpha=0.5)
plt.title('BMI聚类结果及孕周分布 (v1.2)')
plt.xlabel('BMI'); plt.ylabel('孕周 (周)')
plt.legend(title='BMI聚类')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_bmi_clustering_v1.2.png'), dpi=500)
plt.close()

plt.figure(figsize=(12, 6))
clusters = list(cluster_optimal_times.keys())
optimal_times = [cluster_optimal_times[c]['推荐NIPT时点 (周)'] for c in clusters]
bmi_ranges_str = [f"聚类 {c}\n({cluster_bmi_ranges[c][0]:.1f}-{cluster_bmi_ranges[c][1]:.1f})" for c in clusters]
success_rates_pct = [cluster_optimal_times[c]['达标比例'] * 100 for c in clusters]

fig, ax1 = plt.subplots(figsize=(12, 7))
bar1 = ax1.bar(bmi_ranges_str, optimal_times, label='推荐NIPT时点 (周)', color='skyblue')
ax1.set_xlabel('BMI聚类及范围'); ax1.set_ylabel('推荐NIPT时点 (周)', color='darkblue')
ax1.tick_params(axis='y', labelcolor='darkblue'); ax1.set_ylim(bottom=10)

ax2 = ax1.twinx()
line1 = ax2.plot(bmi_ranges_str, success_rates_pct, 'o-', color='green', label='达标比例 (%)')
ax2.set_ylabel('达标比例 (%)', color='green'); ax2.tick_params(axis='y', labelcolor='green'); ax2.set_ylim(0, 105)

lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center')

plt.title('各BMI聚类的推荐NIPT时点和达标比例 (v1.2)')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T2_optimal_nipt_times_v1.2.png'), dpi=500)
plt.close()

# 6. 保存结果到Excel和总结文件
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
    })

results_df = pd.DataFrame(results_data)
excel_path = os.path.join(results_dir, 'T2_bmi_grouping_results_v1.2.xlsx')
results_df.to_excel(excel_path, index=False)

summary_output.append("\n=== 结果汇总 (v1.2) ===")
summary_output.append(f"模型选择: {method_name} | 聚类数: {selected_k} | 轮廓系数: {best_score_val:.4f}")
summary_output.append(f"结果表: {excel_path}")

final_summary = "\n".join(summary_output)
summary_file_path = os.path.join(results_dir, 'T2_analysis_summary_v1.2.txt')
with open(summary_file_path, 'w', encoding='utf-8') as f:
    f.write(final_summary)

print(f"\nT2 v1.2 分析完成！详见: {summary_file_path}")