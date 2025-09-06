"""
任务2-增强分析脚本 (v1.2)
结合了任务3讨论的先进建模思路，用于解决任务2的核心问题。

核心思路:
1.  **目标**: 解决“根据男胎孕妇的BMI，给出合理分组以及每组的最佳NIPT时点，使得孕妇潜在风险最小”的问题。
2.  **建模**: 使用随机森林分类器预测在特定孕周，Y染色体浓度是否能达到4%的标准。
    -   模型考虑了孕妇年龄、BMI、孕周等多个因素。
    -   通过设置 `class_weight='balanced'` 来处理数据不平衡问题。
3.  **风险函数**: 定义一个量化潜在风险的函数，综合考虑了“等待检测的时间成本”和“检测失败（浓度不足）的风险成本”。
    -   Risk(week) = Cost_Wait * (week - min_week) + Cost_Failure * P(Failure at week)
4.  **分组与优化**:
    -   使用K-Means算法对孕妇的BMI进行分组。
    -   对每个BMI分组，模拟计算从第9周到第20周每一周的平均风险。
    -   找到每个组平均风险最低的那一周，作为该组的“最佳NIPT时点”。
5.  **误差分析**: 通过向Y染色体浓度数据中注入模拟噪声，重新进行优化，分析最佳时点的变化，以评估模型的稳健性。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import warnings

warnings.filterwarnings('ignore')

class AdvancedT2Analysis:
    def __init__(self, project_root_path, results_dir_name='results_t3_v1.2'):
        # 设置路径
        self.project_root = project_root_path
        self.results_dir = os.path.join(self.project_root, results_dir_name)
        os.makedirs(self.results_dir, exist_ok=True)

        # 初始化变量
        self.male_data = None
        self.model = None
        self.features = ['孕妇年龄', '孕妇BMI指标', '孕周数值']
        self.target = 'Y_达标'
        
        # 风险函数参数 (可根据实际情况调整)
        self.cost_wait = 0.1  # 每周等待成本
        self.cost_failure = 1.0 # 检测失败成本

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
        """加载和预处理数据，专注于男胎样本"""
        print("--- 1. 加载和预处理数据 ---")
        df = pd.read_csv(data_path, header=None)
        
        # 设置列名
        columns = ['样本序号', '孕妇代码', '孕妇年龄', '孕妇身高', '孕妇体重', '末次月经时间',
                   'IVF妊娠方式', '检测时间', '检测抽血次数', '孕妇本次检测时的孕周', '孕妇BMI指标',
                   '原始测序数据的总读段数', '总读段数中在参考基因组上比对的比例', '总读段数中重复读段的比例',
                   '总读段数中唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值',
                   '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度',
                   'X染色体浓度', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
                   '被过滤掉的读段数占总读段数的比例', '检测出的染色体异常', '孕妇的怀孕次数',
                   '孕妇的生产次数', '胎儿是否健康']
        df.columns = columns

        # 数据清洗和转换
        numeric_cols = ['孕妇年龄', '孕妇BMI指标', 'Y染色体浓度']
        for col in numeric_cols:
            df[col] = df[col].apply(self._safe_float_convert)
        
        df['孕周数值'] = df['孕妇本次检测时的孕周'].apply(self._convert_gestational_age)

        # 筛选男胎数据
        self.male_data = df[(df['Y染色体浓度'].notna()) & (df['Y染色体浓度'] > 0)].copy()
        
        # 定义目标变量：Y染色体浓度是否达到4%
        self.male_data[self.target] = (self.male_data['Y染色体浓度'] >= 0.04).astype(int)
        
        # 移除缺失值
        self.male_data.dropna(subset=self.features + [self.target], inplace=True)
        
        print(f"数据加载完成。男胎样本数: {len(self.male_data)}")
        print(f"Y染色体浓度达标样本比例: {self.male_data[self.target].mean():.2%}")
        return self

    def train_model(self):
        """训练一个分类模型来预测Y染色体浓度是否达标"""
        print("\n--- 2. 训练预测模型 ---")
        X = self.male_data[self.features]
        y = self.male_data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        # 使用随机森林，并处理类别不平衡
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_train, y_train)

        # 评估模型
        y_pred = self.model.predict(X_test)
        print("模型评估报告:")
        print(classification_report(y_test, y_pred))
        return self

    def bmi_grouping(self, n_groups=3):
        """使用K-Means对孕妇BMI进行分组"""
        print(f"\n--- 3. 对BMI进行聚类分组 (k={n_groups}) ---")
        kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        self.male_data['BMI_Group'] = kmeans.fit_predict(self.male_data[['孕妇BMI指标']])
        
        # 调整组标签，使其与BMI大小对应
        group_means = self.male_data.groupby('BMI_Group')['孕妇BMI指标'].mean().sort_values().index
        group_mapping = {old_label: new_label for new_label, old_label in enumerate(group_means)}
        self.male_data['BMI_Group'] = self.male_data['BMI_Group'].map(group_mapping)

        print("各BMI分组的统计信息:")
        print(self.male_data.groupby('BMI_Group')['孕妇BMI指标'].describe())
        return self

    def _calculate_risk(self, week, prob_success):
        """计算给定周数和成功概率下的风险值"""
        prob_failure = 1 - prob_success
        risk = self.cost_wait * (week - 9) + self.cost_failure * prob_failure
        return risk

    def find_optimal_times(self):
        """为每个BMI分组寻找最佳NIPT时点"""
        print("\n--- 4. 为各分组寻找最佳NIPT时点 ---")
        
        optimal_results = {}
        all_risk_curves = {}
        
        for group_id in sorted(self.male_data['BMI_Group'].unique()):
            group_data = self.male_data[self.male_data['BMI_Group'] == group_id]
            group_base_features = group_data[self.features].drop(columns=['孕周数值'])
            
            risk_curve = {}
            for week in range(9, 21): # 模拟第9周到第20周
                # 准备该周的特征数据
                week_features = group_base_features.copy()
                week_features['孕周数值'] = week
                
                # 预测该周的成功概率
                prob_success = self.model.predict_proba(week_features)[:, 1]
                
                # 计算该周的平均风险
                avg_risk = self._calculate_risk(week, prob_success.mean())
                risk_curve[week] = avg_risk

            # 找到风险最低的周
            optimal_week = min(risk_curve, key=risk_curve.get)
            min_risk = risk_curve[optimal_week]
            
            bmi_range = (group_data['孕妇BMI指标'].min(), group_data['孕妇BMI指标'].max())

            optimal_results[group_id] = {
                'Optimal_Week': optimal_week,
                'Min_Risk': min_risk,
                'BMI_Range': f"{bmi_range[0]:.1f} - {bmi_range[1]:.1f}",
                'Sample_Count': len(group_data)
            }
            all_risk_curves[group_id] = risk_curve
        
        # 打印和保存结果
        report = []
        report.append("="*40)
        report.append("最佳NIPT时点分析结果")
        report.append("="*40)
        for group_id, result in optimal_results.items():
            line = (f"BMI分组 {group_id} (BMI范围: {result['BMI_Range']}, 样本数: {result['Sample_Count']}): "
                    f"最佳检测周为 {result['Optimal_Week']}，"
                    f"最低平均风险值为 {result['Min_Risk']:.3f}")
            print(line)
            report.append(line)
        report.append("="*40)
        
        with open(os.path.join(self.results_dir, 'T2_optimal_timing_report.txt'), 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
            
        self.plot_risk_curves(all_risk_curves, optimal_results)
        
        return optimal_results, all_risk_curves

    def plot_risk_curves(self, all_risk_curves, optimal_results):
        """可视化每个组的风险曲线"""
        plt.figure(figsize=(12, 7))
        
        for group_id, risk_curve in all_risk_curves.items():
            weeks = list(risk_curve.keys())
            risks = list(risk_curve.values())
            bmi_range = optimal_results[group_id]['BMI_Range']
            label = f"BMI分组 {group_id} (范围: {bmi_range})"
            plt.plot(weeks, risks, marker='o', linestyle='-', label=label)

        plt.title('各BMI分组的NIPT检测潜在风险曲线', fontsize=16)
        plt.xlabel('孕周', fontsize=12)
        plt.ylabel('平均潜在风险值', fontsize=12)
        plt.xticks(range(9, 21))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        
        fig_path = os.path.join(self.results_dir, 'T2_risk_curves_by_bmi_group.png')
        plt.savefig(fig_path, dpi=300)
        print(f"\n风险曲线图已保存至: {fig_path}")
        plt.close()

    def error_sensitivity_analysis(self, noise_level=0.005):
        """分析检测误差对结果的影响"""
        print(f"\n--- 5. 误差敏感性分析 (噪声水平: {noise_level}) ---")
        
        # 复制原始数据并添加噪声
        data_with_noise = self.male_data.copy()
        noise = np.random.normal(0, noise_level, data_with_noise.shape[0])
        data_with_noise['Y染色体浓度'] += noise
        
        # 重新定义目标变量
        data_with_noise[self.target] = (data_with_noise['Y染色体浓度'] >= 0.04).astype(int)
        data_with_noise.dropna(subset=self.features + [self.target], inplace=True)

        # 使用带噪声的数据重新训练模型
        X_noise = data_with_noise[self.features]
        y_noise = data_with_noise[self.target]
        noise_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        noise_model.fit(X_noise, y_noise)
        
        # 重新寻找最佳时点
        print("在模拟检测误差下，重新计算最佳时点:")
        original_model = self.model
        self.model = noise_model # 临时替换模型
        
        noisy_optimal_results, _ = self.find_optimal_times()
        
        self.model = original_model # 恢复原始模型
        
        # 比较结果
        print("\n误差影响对比:")
        # (此处可以添加更详细的对比报告)
        
        return noisy_optimal_results

    def run_complete_analysis(self):
        """执行完整的分析流程"""
        data_path = os.path.join(self.project_root, 'Source_DATA', 'dataA.csv')
        self.load_and_preprocess_data(data_path)
        self.train_model()
        self.bmi_grouping(n_groups=3)
        self.find_optimal_times()
        self.error_sensitivity_analysis()
        print("\n--- 分析完成 ---")


if __name__ == '__main__':
    # 获取项目根目录 (假设此脚本在 CUMCM2025-C/t3_scripts_v1.2/ 目录下)
    current_script_path = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(current_script_path)
    project_root = os.path.dirname(scripts_dir)

    analyzer = AdvancedT2Analysis(project_root_path=project_root)
    analyzer.run_complete_analysis()