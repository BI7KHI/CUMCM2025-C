# -*- coding: utf-8 -*-
"""
任务3 增强分析脚本 v1.4 (深度优化)

核心改进:
1.  **SMOTE过采样**: 解决数据不平衡问题，提升对少数类（不达标）的识别能力。
2.  **动态风险成本**: 引入随孕周增加而增长的失败成本，使风险评估更贴近现实。
3.  **结构化与模块化**: 优化代码结构，便于维护和扩展。
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 全局化配置 ---
# 设置字体，确保图表中文和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义文件路径
BASE_DIR = r"C:\Users\Admin\Desktop\CUMCM2025-C\CUMCM2025-C"
DATA_FILE = os.path.join(BASE_DIR, "Processed_DATA", "cleaned_male_fetal_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "T3", "beta", "v1.4")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# 定义模型和分析参数
FEATURES = ['孕周数值', '孕妇年龄', '孕妇BMI指标']
TARGET = 'Y染色体浓度是否达标'
TEST_SIZE = 0.3
RANDOM_STATE = 42
WEEKS_RANGE = range(9, 21) # 孕周范围 9-20

# --- 2. 数据加载与预处理 ---
def load_and_prepare_data(file_path):
    """加载数据并进行基础预处理"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # 填充缺失值
    for col in FEATURES:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"列 '{col}' 中的缺失值已用中位数 ({median_val:.2f}) 填充。")

    print("--- Debug: DataFrame Columns ---")
    print(df.columns)
    print("---------------------------------")
    df[TARGET] = df['Y染色体浓度'].apply(lambda x: 1 if x >= 2.5 else 0)
    print("数据加载完成，预处理完毕。")
    print(f"Y染色体浓度达标（1）与不达标（0）样本分布:\n{df[TARGET].value_counts(normalize=True)}")
    return df

# --- 3. 模型训练与评估 (集成SMOTE) ---
def train_evaluate_model(df):
    """使用SMOTE处理不平衡数据并训练随机森林模型"""
    print("--- Debug: Columns in train_evaluate_model ---")
    print(df.columns)
    print("---------------------------------------------")
    X = df[FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    # 动态调整SMOTE的k_neighbors参数
    n_minority_samples = pd.Series(y_train).value_counts().min()
    if n_minority_samples <= 5:
        k_neighbors = max(1, n_minority_samples - 1)
        print(f"\n少数类样本不足({n_minority_samples})，SMOTE的k_neighbors被动态调整为{k_neighbors}。")
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    else:
        smote = SMOTE(random_state=RANDOM_STATE)

    # 应用SMOTE过采样
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("\n应用SMOTE后，训练集样本分布:")
    print(pd.Series(y_train_resampled).value_counts(normalize=True))
    
    # 超参数调优
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_weighted')
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    best_model = grid_search.best_estimator_
    print(f"\n最佳模型参数: {grid_search.best_params_}")
    
    # 在原始测试集上评估
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['不达标', '达标'])
    
    report_path = os.path.join(RESULTS_DIR, "T3_beta_v1.4.4.txt")
    with open(report_path, "w", encoding='utf-8-sig') as f:
        f.write("模型评估报告 (SMOTE + GridSearchCV):\n")
        f.write(report)
        f.write("\n混淆矩阵:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
        
    print("\n模型评估报告已保存。")
    print(report)
    
    return best_model

# --- 4. BMI动态分组 ---
def dynamic_bmi_clustering(df):
    """使用轮廓系数确定最佳BMI分组数并进行聚类"""
    bmi_data = df[['孕妇BMI指标']].values
    scaler = StandardScaler()
    bmi_scaled = scaler.fit_transform(bmi_data)
    
    silhouette_scores = []
    cluster_range = range(2, 6)
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(bmi_scaled)
        silhouette_avg = silhouette_score(bmi_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
    
    # 绘制轮廓系数图
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title('BMI聚类轮廓系数分析')
    plt.xlabel('聚类数量 (k)')
    plt.ylabel('轮廓系数')
    plt.xticks(cluster_range)
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'T3_beta_v1.4.4.png'))
    plt.close()
    
    print(f"\n根据轮廓系数，最佳BMI分组数为: {best_n_clusters}")
    
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=RANDOM_STATE, n_init=10)
    df['BMI_Group'] = kmeans.fit_predict(bmi_scaled)
    
    # 显式地将标准化的BMI值还原，确保后续报告使用的是原始值
    df['孕妇BMI指标'] = scaler.inverse_transform(bmi_scaled)
    
    # 分析并保存分组信息
    group_stats = df.groupby('BMI_Group')['孕妇BMI指标'].agg(['mean', 'min', 'max', 'count']).sort_values('mean')
    report_path = os.path.join(RESULTS_DIR, "T3_beta_v1.4.4.txt")
    with open(report_path, "w", encoding='utf-8-sig') as f:
        f.write("BMI 动态分组统计信息:\n")
        f.write(str(group_stats))
    print("BMI分组完成，统计信息已保存。")
    print(group_stats)
    
    return df, group_stats.index.tolist()

# --- 5. 风险函数与时点优化 (动态成本) ---
def calculate_dynamic_risk(prob_success, week, cost_wait=0.1, cost_failure_base=1.0, cost_increase_rate=0.1):
    """
    计算具有动态失败成本的风险函数。
    - prob_success: 在给定孕周检测成功的概率。
    - week: 当前孕周。
    - cost_wait: 每周的等待成本。
    - cost_failure_base: 基础失败成本。
    - cost_increase_rate: 失败成本每周增长率。
    """
    cost_failure_dynamic = cost_failure_base * (1 + cost_increase_rate * (week - min(WEEKS_RANGE)))
    expected_cost = (week - min(WEEKS_RANGE)) * cost_wait + (1 - prob_success) * cost_failure_dynamic
    return expected_cost

def find_optimal_timing(model, bmi_group_df, group_id):
    """为单个BMI分组寻找最佳检测时点"""
    risks = []
    mean_age = bmi_group_df['孕妇年龄'].mean()
    mean_bmi = bmi_group_df['孕妇BMI指标'].mean()
    
    for week in WEEKS_RANGE:
        # 创建用于预测的单一样本
        sample = pd.DataFrame([[week, mean_age, mean_bmi]], columns=FEATURES)
        prob_success = model.predict_proba(sample)[0, 1]
        risk = calculate_dynamic_risk(prob_success, week)
        risks.append(risk)
        
    optimal_week = WEEKS_RANGE[np.argmin(risks)]
    min_risk = np.min(risks)
    
    return optimal_week, min_risk, risks

# --- 6. 主流程与结果生成 ---
def main():
    """执行完整的分析流程"""
    # 步骤1: 加载数据
    df = load_and_prepare_data(DATA_FILE)
    
    # 步骤2: 训练模型
    model = train_evaluate_model(df)
    
    # 步骤3: BMI分组
    df_grouped, group_ids = dynamic_bmi_clustering(df)
    
    # 步骤4: 寻找最佳时点并可视化
    plt.figure(figsize=(12, 8))
    report_content = "深度优化后的最佳NIPT检测时点报告 (v1.4)\n"
    report_content += "="*50 + "\n"
    
    for group_id in group_ids:
        group_df = df_grouped[df_grouped['BMI_Group'] == group_id]
        group_bmi_range = f"{group_df['孕妇BMI指标'].min():.2f}-{group_df['孕妇BMI指标'].max():.2f}"
        
        optimal_week, min_risk, risks = find_optimal_timing(model, group_df, group_id)
        
        # 记录报告
        report_content += f"\nBMI 分组 {group_id} (范围: {group_bmi_range}):\n"
        report_content += f"  - 最佳检测孕周: {optimal_week} 周\n"
        report_content += f"  - 最低风险值: {min_risk:.4f}\n"
        
        # 绘制风险曲线
        plt.plot(list(WEEKS_RANGE), risks, marker='o', label=f'BMI组 {group_id} ({group_bmi_range})')

    plt.title('各BMI分组的NIPT检测风险曲线 (动态成本模型)')
    plt.xlabel('孕周')
    plt.ylabel('预期风险值')
    plt.xticks(list(WEEKS_RANGE))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(RESULTS_DIR, 'T3_beta_v1.4.4.png'))
    plt.close()
    
    # 保存最终报告
    report_path = os.path.join(RESULTS_DIR, "T3_beta_v1.4.4.txt")
    with open(report_path, "w", encoding='utf-8-sig') as f:
        f.write(report_content)
        
    print("\n分析完成，风险曲线和最终报告已保存至:")
    print(RESULTS_DIR)

if __name__ == "__main__":
    main()