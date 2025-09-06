"""
任务2-超参数调优与动态分组分析脚本 (v1.3)
在v1.2的基础上进行深度优化。

核心改进:
1.  **动态BMI分组**: 不再固定K=3，而是通过计算轮廓系数(Silhouette Score)来动态寻找最佳的聚类数量，使得分组更具数据驱动性。
2.  **模型超参数调优**: 使用`GridSearchCV`对随机森林分类器进行系统性的超参数搜索，找到最优的模型配置，以提升预测Y染色体浓度达标可能性的准确率和鲁棒性。
3.  **结构化代码**: 将分析流程封装在类中，提高代码的可读性和可维护性。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, silhouette_score
import os
import warnings

warnings.filterwarnings('ignore')

class OptimizedT2Analysis:
    def __init__(self, project_root_path, results_dir_name='results_t3_v1.3_optimized'):
        # 设置路径
        self.project_root = project_root_path
        self.results_dir = os.path.join(self.project_root, results_dir_name)
        os.makedirs(self.results_dir, exist_ok=True)

        # 初始化变量
        self.male_data = None
        self.model = None
        self.features = ['孕妇年龄', '孕妇BMI指标', '孕周数值']
        self.target = 'Y_达标'
        
        # 风险函数参数
        self.cost_wait = 0.1
        self.cost_failure = 1.0

        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

    def _safe_float_convert(self, x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan

    def _convert_gestational_age(self, age_str):
        try:
            if isinstance(age_str, str):
                if 'w+' in age_str:
                    weeks, days = age_str.split('w+')
                    return float(weeks) + float(days) / 7
                elif 'w' in age_str:
                    return float(age_str.split('w')[0])
            return float(age_str)
        except (ValueError, TypeError):
            return np.nan

    def load_and_preprocess_data(self, data_path):
        print("--- 1. 加载和预处理数据 ---")
        df = pd.read_csv(data_path, header=None)
        # ... (此处省略与v1.2相同的列名设置和基础清洗代码) ...
        columns = ['样本序号', '孕妇代码', '孕妇年龄', '孕妇身高', '孕妇体重', '末次月经时间',
                   'IVF妊娠方式', '检测时间', '检测抽血次数', '孕妇本次检测时的孕周', '孕妇BMI指标',
                   '原始测序数据的总读段数', '总读段数中在参考基因组上比对的比例', '总读段数中重复读段的比例',
                   '总读段数中唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值',
                   '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度',
                   'X染色体浓度', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
                   '被过滤掉的读段数占总读段数的比例', '检测出的染色体异常', '孕妇的怀孕次数',
                   '孕妇的生产次数', '胎儿是否健康']
        df.columns = columns
        numeric_cols = ['孕妇年龄', '孕妇BMI指标', 'Y染色体浓度']
        for col in numeric_cols:
            df[col] = df[col].apply(self._safe_float_convert)
        df['孕周数值'] = df['孕妇本次检测时的孕周'].apply(self._convert_gestational_age)
        self.male_data = df[(df['Y染色体浓度'].notna()) & (df['Y染色体浓度'] > 0)].copy()
        self.male_data[self.target] = (self.male_data['Y染色体浓度'] >= 0.04).astype(int)
        self.male_data.dropna(subset=self.features + [self.target], inplace=True)
        print(f"数据加载完成。男胎样本数: {len(self.male_data)}")
        return self

    def tune_and_train_model(self):
        print("\n--- 2. 模型超参数调优与训练 ---")
        X = self.male_data[self.features]
        y = self.male_data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        # 设置超参数搜索空间
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        # 使用GridSearchCV进行超参数搜索
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1_weighted')
        grid_search.fit(X_train, y_train)

        print(f"最佳超参数: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_

        # 在测试集上评估最终模型
        y_pred = self.model.predict(X_test)
        print("\n最终模型评估报告:")
        print(classification_report(y_test, y_pred))
        return self

    def find_optimal_clusters(self, max_k=8):
        print(f"\n--- 3. 动态寻找最佳BMI聚类数量 ---")
        bmi_data = self.male_data[['孕妇BMI指标']]
        scaler = StandardScaler()
        bmi_scaled = scaler.fit_transform(bmi_data)

        silhouette_scores = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(bmi_scaled)
            score = silhouette_score(bmi_scaled, labels)
            silhouette_scores.append(score)
            print(f"聚类数 K={k}, 轮廓系数: {score:.4f}")

        # 找到最佳聚类数
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\n最佳聚类数量确定为: K={optimal_k}")

        # 可视化轮廓系数
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, marker='o')
        plt.title('K-Means聚类轮廓系数变化图', fontsize=16)
        plt.xlabel('聚类数量 (K)', fontsize=12)
        plt.ylabel('轮廓系数', fontsize=12)
        plt.xticks(k_range)
        plt.grid(True)
        fig_path = os.path.join(self.results_dir, 'T3_silhouette_scores_for_bmi.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"轮廓系数图已保存至: {fig_path}")

        return optimal_k

    def bmi_grouping(self, n_groups):
        print(f"\n--- 4. 对BMI进行聚类分组 (K={n_groups}) ---")
        kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        self.male_data['BMI_Group'] = kmeans.fit_predict(self.male_data[['孕妇BMI指标']])
        
        group_means = self.male_data.groupby('BMI_Group')['孕妇BMI指标'].mean().sort_values().index
        group_mapping = {old_label: new_label for new_label, old_label in enumerate(group_means)}
        self.male_data['BMI_Group'] = self.male_data['BMI_Group'].map(group_mapping)

        print("各BMI分组的统计信息:")
        print(self.male_data.groupby('BMI_Group')['孕妇BMI指标'].describe())
        return self

    def _calculate_risk(self, week, prob_success):
        prob_failure = 1 - prob_success
        risk = self.cost_wait * (week - 9) + self.cost_failure * prob_failure
        return risk

    def find_optimal_times(self):
        print("\n--- 5. 为各分组寻找最佳NIPT时点 ---")
        # ... (此部分与v1.2版本相同，此处省略) ...
        optimal_results = {}
        all_risk_curves = {}
        for group_id in sorted(self.male_data['BMI_Group'].unique()):
            group_data = self.male_data[self.male_data['BMI_Group'] == group_id]
            group_base_features = group_data[self.features].drop(columns=['孕周数值'])
            risk_curve = {}
            for week in range(9, 21):
                week_features = group_base_features.copy()
                week_features['孕周数值'] = week
                prob_success = self.model.predict_proba(week_features)[:, 1]
                avg_risk = self._calculate_risk(week, prob_success.mean())
                risk_curve[week] = avg_risk
            optimal_week = min(risk_curve, key=risk_curve.get)
            min_risk = risk_curve[optimal_week]
            bmi_range = (group_data['孕妇BMI指标'].min(), group_data['孕妇BMI指标'].max())
            optimal_results[group_id] = {
                'Optimal_Week': optimal_week, 'Min_Risk': min_risk,
                'BMI_Range': f"{bmi_range[0]:.1f} - {bmi_range[1]:.1f}",
                'Sample_Count': len(group_data)
            }
            all_risk_curves[group_id] = risk_curve
        report = ["="*40, "最佳NIPT时点分析结果 (优化后)", "="*40]
        for group_id, result in optimal_results.items():
            line = (f"BMI分组 {group_id} (BMI: {result['BMI_Range']}, N={result['Sample_Count']}): "
                    f"最佳检测周: {result['Optimal_Week']}, 最低风险: {result['Min_Risk']:.3f}")
            print(line)
            report.append(line)
        report.append("="*40)
        with open(os.path.join(self.results_dir, 'T3_optimal_timing_report_optimized.txt'), 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        self.plot_risk_curves(all_risk_curves, optimal_results)
        return self

    def plot_risk_curves(self, all_risk_curves, optimal_results):
        plt.figure(figsize=(12, 7))
        for group_id, risk_curve in all_risk_curves.items():
            label = f"BMI分组 {group_id} ({optimal_results[group_id]['BMI_Range']})"
            plt.plot(list(risk_curve.keys()), list(risk_curve.values()), marker='o', label=label)
        plt.title('各BMI分组的NIPT检测潜在风险曲线 (优化后)', fontsize=16)
        plt.xlabel('孕周', fontsize=12)
        plt.ylabel('平均潜在风险值', fontsize=12)
        plt.xticks(range(9, 21))
        plt.grid(True, linestyle='--')
        plt.legend()
        fig_path = os.path.join(self.results_dir, 'T3_risk_curves_optimized.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"\n风险曲线图已保存至: {fig_path}")

    def run_complete_analysis(self):
        """执行完整的优化分析流程"""
        data_path = os.path.join(self.project_root, 'Source_DATA', 'dataA.csv')
        self.load_and_preprocess_data(data_path)
        self.tune_and_train_model()
        optimal_k = self.find_optimal_clusters()
        self.bmi_grouping(n_groups=optimal_k)
        self.find_optimal_times()
        print("\n--- 优化分析完成 ---")

if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    # 假设脚本在 CUMCM2025-C/t3_scripts_v1.3/ 目录下
    project_root = os.path.dirname(os.path.dirname(current_script_path))

    analyzer = OptimizedT2Analysis(project_root_path=project_root)
    analyzer.run_complete_analysis()