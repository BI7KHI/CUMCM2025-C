# T3题目分析说明

## 题目描述
分析X染色体浓度异常对NIPT检测结果的影响，建立数学模型进行异常检测和风险评估。

## 文件结构

### 脚本文件 (t3_scripts_v1.0/)
- `t3_analysis.py` - T3原版分析脚本，包含机器学习分类和异常检测
- `t3_enhanced_analysis.py` - T3增强版分析脚本，结合T1统计建模和T2聚类分析思路

### 结果文件
- `results_t3_v1.0/` - 原版T3分析结果
- `results_t3_v1.1_enhanced/` - 增强版T3分析结果

## 分析方法对比

### 原版T3分析 (t3_analysis.py)
- **核心方法**: 机器学习分类 (Random Forest, SVM, Logistic Regression等)
- **异常检测**: 基于统计阈值的方法
- **特征工程**: 基础的数据预处理
- **输出结果**: 分类报告、混淆矩阵、ROC/PR曲线、特征重要性

### 增强版T3分析 (t3_enhanced_analysis.py)  
- **核心方法**: 统计建模 + 聚类分析 + 机器学习
- **统计分析**: VIF共线性诊断、逻辑回归、假设检验
- **聚类分析**: K-means患者分群、轮廓系数优化
- **异常检测**: 多种统计方法(IQR + 均值±2σ)
- **风险评估**: 基于聚类的风险分层和个性化临床建议

## 运行方法

### 原版分析
```bash
cd /path/to/CUMCM2025-C
python3 t3_scripts_v1.0/t3_analysis.py
```

### 增强版分析  
```bash
cd /path/to/CUMCM2025-C
python3 t3_scripts_v1.0/t3_enhanced_analysis.py
```

## 依赖包
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0
- openpyxl (for Excel output)

## 主要创新点

### 1. 方法论融合
- 借鉴T1的统计建模思路：VIF诊断、逻辑回归、假设检验
- 借鉴T2的聚类分析思路：K-means分群、轮廓系数优化
- 结合机器学习：多模型对比、超参数优化

### 2. 临床应用导向
- 患者分群：发现不同风险亚群
- 个性化建议：基于聚类的分层管理
- 可解释性：提供生物学意义的解释

### 3. 技术改进
- 多重异常检测：IQR + 统计学双重标准
- 字体兼容性：解决中文显示问题，使用国际通用英文标签
- 完整性：从数据预处理到结果可视化的完整流程

## 结果解读

### 原版T3主要发现
- 识别出X染色体浓度异常样本
- Random Forest模型表现最佳
- 提供特征重要性排序

### 增强版T3主要发现
- **生物学发现**: 孕周是X染色体浓度异常的显著影响因素 (p=0.014)
- **患者分群**: 发现早期高浓度vs晚期低浓度两个亚群  
- **风险分层**: 不同亚群异常风险存在显著差异
- **临床意义**: 支持个性化检测策略

## 文件清单

### 原版结果 (results_t3_v1.0/)
- T3_x_chromosome_distribution.png - X染色体浓度分布图
- T3_feature_importance_Random Forest.png - 特征重要性图
- T3_roc_curve_Random Forest_optimized.png - ROC曲线
- T3_pr_curve_Random Forest_optimized.png - PR曲线
- T3_correlation_heatmap.png - 相关性热图
- T3_model_evaluation_results.xlsx - 模型评估结果
- T3_correlation_matrix.xlsx - 相关性矩阵
- T3_feature_importance.xlsx - 特征重要性数据

### 增强版结果 (results_t3_v1.1_enhanced/)
- T3_clustering_analysis.png - 聚类分析可视化
- T3_comprehensive_analysis.png - 综合分析图表
- T3_logistic_regression_results.txt - 逻辑回归详细结果
- T3_enhanced_analysis_report.md - 完整分析报告

## 版本历史
- v1.0: 基础机器学习分类分析
- v1.1_enhanced: 增强版分析，融合统计建模和聚类分析思路

## 联系信息
Created for CUMCM 2025 Contest - Problem C
Date: September 2025
