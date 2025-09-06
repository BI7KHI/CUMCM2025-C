# T3 分析脚本 v1.4 (深度优化) 说明文档

## 1. 概述

`t3_analysis_v1.4.py` 脚本是为解决 “Y染色体浓度不达标” 预测问题而开发的深度优化版本。此版本在前序工作的基础上，引入了更高级的数据处理技术和动态风险评估策略，旨在显著提升模型的预测性能和决策的精准性。

核心优化点包括：

- **处理数据不平衡**：引入 **SMOTE (Synthetic Minority Over-sampling Technique)** 算法，通过合成少数类样本来解决训练数据中“达标”与“不达标”样本比例严重失衡的问题。
- **动态风险成本**：将风险函数中的失败成本（`Cost_fp`）从一个固定值调整为随孕周动态增长的函数，更真实地反映了随孕周增加，误判（假阴性）带来的风险和成本越高的实际情况。
- **健壮性与自动化**：
    - 实现了对 SMOTE 算法 `k_neighbors` 参数的动态调整，避免了因交叉验证折中少数类样本过少而导致的程序崩溃。
    - 增加了缺失值处理步骤，使用中位数填充关键特征的缺失数据，增强了模型的稳定性。
    - 统一了项目文件的编码格式为 `utf-8-sig`，解决了在不同环境下（特别是Windows）因BOM头导致的读取错误问题。

## 2. 脚本核心功能

### 2.1. 数据加载与预处理 (`load_and_prepare_data`)

- **加载数据**: 从 `cleaned_male_fetal_data.csv` 文件加载数据。
- **编码格式**: 明确使用 `utf-8-sig` 编码，以兼容带BOM的UTF-8文件。
- **缺失值填充**: 对 `FEATURES` 列表中的关键特征（'孕周数值', '孕妇年龄', '孕妇BMI指标'）使用各自列的中位数进行填充，确保数据完整性。

### 2.2. 模型训练与评估 (`train_evaluate_model`)

- **SMOTE 过采样**: 在交叉验证的每个训练折中，应用 SMOTE 算法来平衡正负样本。
- **动态 `k_neighbors`**: 在应用 SMOTE 前，检查少数类样本数量。如果样本数不足（小于等于5），则动态调整 `k_neighbors` 参数为 `n_samples - 1`，保证算法能够顺利运行。
- **GridSearchCV**: 使用网格搜索和5折交叉验证，在给定的参数网格中寻找随机森林分类器的最佳超参数组合。
- **模型评估**: 使用最佳模型在测试集上进行评估，并生成包含精确率、召回率、F1-score的分类报告，保存为 `T3_model_evaluation_report_SMOTE.txt`。

### 2.3. BMI 动态聚类 (`dynamic_bmi_clustering`)

- **KMeans聚类**: 基于标准化的孕妇BMI数据，使用KMeans算法进行聚类。
- **轮廓系数**: 通过计算不同聚类数（2至5组）的轮廓系数，自动确定最佳的BMI分组数量。
- **结果保存**: 将聚类结果（BMI_Group）添加回主DataFrame，并保存分组的统计信息（均值、最大/小值、样本数）到 `T3_bmi_group_report.txt`。

### 2.4. 风险评估与最佳时点决策 (`calculate_risk`, `find_best_decision_point`)

- **动态失败成本**: 定义 `cost_fp(week)` 函数，使得假阴性预测的成本随孕周（`week`）线性增长。公式为 `5000 + 1000 * (week - 10)`。
- **风险计算**: `calculate_risk` 函数基于模型预测概率、动态成本函数以及检测成本，计算在特定孕周进行检测的总期望风险。
- **寻找最佳时点**: `find_best_decision_point` 函数遍历所有指定的孕周（10-24周），计算每个孕周的总风险，并找出风险最低的孕周作为最佳决策时点。
- **结果输出**: 
    - 将每个BMI分组在不同孕周的风险评估结果保存到 `T3_risk_assessment_results_SMOTE.csv`。
    - 生成最终的决策报告 `T3_optimal_timing_report_deep_optimized.txt`，清晰地列出每个BMI分组的最佳检测孕周、对应的最低风险和BMI范围。

## 3. 如何运行

1. 确保已安装所有必需的Python库，特别是 `pandas`, `scikit-learn`, `imbalanced-learn`。
2. 将脚本 `t3_analysis_v1.4.py` 与数据文件 `cleaned_male_fetal_data.csv` 放置在正确的目录结构下。
3. 直接通过命令行运行脚本：
   ```bash
   python t3_analysis_v1.4.py
   ```

## 4. 输出文件

脚本运行成功后，将在 `results_t3_v1.4_deep_optimized` 目录下生成以下文件：

- `T3_model_evaluation_report_SMOTE.txt`: 包含SMOTE优化后模型在测试集上的详细性能指标。
- `T3_bmi_group_report.txt`: BMI动态聚类的统计信息。
- `T3_risk_assessment_results_SMOTE.csv`: 各BMI分组在不同孕周下的详细风险评估数据。
- `T3_optimal_timing_report_deep_optimized.txt`: 最终的结论报告，指明了不同BMI分组的最佳检测时点和对应的风险值。