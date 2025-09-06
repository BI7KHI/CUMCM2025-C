import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score
from sklearn.utils import resample
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

# 读取数据
print("开始读取数据...")
data_path = os.path.join(project_root, 'data', 'common', 'source', 'dataA.csv')
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

# 数据预处理函数
def safe_float_convert(x):
    try:
        return float(x)
    except:
        return np.nan

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

# 数据预处理
print("开始数据预处理...")
# 转换数值型列的数据类型
numeric_columns = ['孕妇年龄', '孕妇身高', '孕妇体重', '孕妇BMI指标',
                   '原始测序数据的总读段数', '总读段数中在参考基因组上比对的比例', '总读段数中重复读段的比例',
                   '总读段数中唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值',
                   '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度',
                   'X染色体浓度', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
                   '被过滤掉的读段数占总读段数的比例']

for col in numeric_columns:
    data[col] = data[col].apply(safe_float_convert)

# 转换孕周为数值
data['孕周数值'] = data['孕妇本次检测时的孕周'].apply(convert_gestational_age)

# 任务T4：确定女胎异常的判定方法

# 1. 筛选女胎数据
print("筛选女胎数据...")
# 女胎：Y染色体浓度为0或空
female_fetus_data = data[(data['Y染色体浓度'].isna()) | (data['Y染色体浓度'] == 0)].copy()
print(f'女胎样本总数: {len(female_fetus_data)}')

# 2. 数据分析前的准备
# 标记染色体异常（非整倍体）
female_fetus_data['染色体异常'] = female_fetus_data['检测出的染色体异常'].notna().astype(int)

# 统计异常样本数
abnormal_count = female_fetus_data['染色体异常'].sum()
normal_count = len(female_fetus_data) - abnormal_count
print(f'女胎异常样本数: {abnormal_count}')
print(f'女胎正常样本数: {normal_count}')

# 3. 相关性分析：分析各因素与女胎染色体异常的关系
print("进行相关性分析...")
# 定义可能的特征列
potential_features = ['孕妇年龄', '孕妇BMI指标', '孕周数值', 'X染色体浓度', 'GC含量',
                      '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
                      '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
                      '原始测序数据的总读段数', '总读段数中在参考基因组上比对的比例', '总读段数中重复读段的比例']

# 计算各特征与染色体异常的相关性
correlation_results = {}
for feature in potential_features:
    valid_data = female_fetus_data[[feature, '染色体异常']].dropna()
    if len(valid_data) > 1:
        corr_coef, p_value = stats.pointbiserialr(valid_data['染色体异常'], valid_data[feature])
        correlation_results[feature] = {'相关系数': corr_coef, 'p值': p_value}
    else:
        correlation_results[feature] = {'相关系数': np.nan, 'p值': np.nan}

# 保存相关性分析结果
correlation_df = pd.DataFrame(correlation_results).T
correlation_df.to_excel(os.path.join(results_dir, 'T4_correlation_results.xlsx'))

print("相关性分析结果:")
print(correlation_df.sort_values('相关系数', ascending=False))

# 可视化相关性热力图
plt.figure(figsize=(12, 10))
valid_vars = ['染色体异常'] + [var for var in potential_features if not np.isnan(correlation_results[var]['相关系数'])]
correlation_matrix = female_fetus_data[valid_vars].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('女胎染色体异常与各因素的相关性热力图')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T4_correlation_heatmap.png'), dpi=300)
plt.close()

# 4. 特征工程与模型建立
print("进行特征工程与模型建立...")
# 定义特征列（基于相关性分析结果选择）
feature_columns = ['孕妇年龄', '孕妇BMI指标', '孕周数值', 'X染色体浓度', 'GC含量',
                   '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
                   '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']

# 查看女胎数据中的染色体异常情况
print("女胎数据中的染色体异常情况统计:")
print(female_fetus_data['检测出的染色体异常'].value_counts(dropna=False))

# 准备模型数据
# 首先只删除染色体异常列的空值
model_data = female_fetus_data.dropna(subset=['染色体异常']).copy()
print(f'删除染色体异常列为空的样本后，剩余样本数: {len(model_data)}')

# 如果样本数仍然为0，尝试使用AB列（非整倍体）作为标签
if len(model_data) == 0:
    print("警告: 没有找到染色体异常列不为空的女胎样本，尝试使用其他方法确定异常标签...")
    # 查看AB列数据
    print("尝试使用原始数据中的AB列信息...")
    # 我们假设AB列对应的是第27列(检测出的染色体异常)
    # 创建一个新的异常标签列，基于原始数据中的第27列
    female_fetus_data['染色体异常'] = female_fetus_data.iloc[:, 27].notna().astype(int)
    model_data = female_fetus_data.copy()
    print(f'新标签下的样本数: {len(model_data)}')
    print(f'异常样本数: {model_data["染色体异常"].sum()}')

# 处理特征列中的缺失值
print(f'特征列中各列的缺失值数量:')
missing_values = model_data[feature_columns].isnull().sum()
print(missing_values)

# 如果样本数仍然很少，我们需要调整分析策略
if len(model_data) < 10:
    print("警告: 女胎样本数量过少，无法进行传统的机器学习建模。")
    print("将采用统计学方法和专家规则进行分析...")
    
    # 直接进行相关性分析和特征重要性分析
    # 计算各特征与染色体异常的相关性（使用完整数据，不进行训练测试拆分）
    print("\n基于专家规则和文献研究的女胎异常判定方法:")
    
    # 由于女胎样本数量极少，我们将结合医学指南和专家共识制定判定规则
    print("\n基于医学指南和专家共识的女胎异常判定方法:")
    
    # 1. 染色体Z值分析（参考医学标准）
    print("\n1. 染色体Z值异常判定标准:")
    # 参考行业标准，通常Z值绝对值大于3被认为是异常
    chrom_z_thresholds = {
        '13号染色体的Z值': 3.0,
        '18号染色体的Z值': 3.0,
        '21号染色体的Z值': 3.0,
        'X染色体的Z值': 3.0  # 对于X染色体异常，如特纳综合征等
    }
    
    # 统计数据中的Z值分布（如果有数据）
    z_stats = []
    for chrom, threshold in chrom_z_thresholds.items():
        if chrom in model_data.columns:
            valid_data = model_data[chrom].dropna()
            if len(valid_data) > 0:
                mean_val = valid_data.mean()
                std_val = valid_data.std()
                abnormal_count = ((valid_data < mean_val - threshold * std_val) | 
                                 (valid_data > mean_val + threshold * std_val)).sum()
                z_stats.append({
                    '染色体': chrom,
                    '均值': mean_val,
                    '标准差': std_val,
                    '异常样本数': abnormal_count,
                    '异常比例': abnormal_count / len(valid_data) if len(valid_data) > 0 else 0
                })
                print(f"   {chrom}: 均值={mean_val:.4f}, 标准差={std_val:.4f}, 异常阈值建议: ±{threshold}倍标准差")
            else:
                print(f"   {chrom}: 无有效数据，采用通用标准: Z值绝对值>{threshold}判定为异常")
    
    # 2. 综合风险评估指标
    print("\n2. 综合风险评估指标:")
    print("   a. 染色体异常标记: 直接使用检测结果中的染色体异常标记")
    print("   b. 多指标异常计数: 计算超出正常范围的指标数量")
    print("   c. 风险评分: 基于各指标异常程度进行加权评分")
    
    # 3. 不同孕周和BMI的风险调整
    print("\n3. 不同孕周和BMI的风险调整:")
    print("   - 早孕期（<12周）: 风险阈值可适当降低，建议结合其他检查")
    print("   - 中孕期（12-27周）: 使用标准阈值")
    print("   - 高BMI孕妇（BMI>30）: 风险阈值可适当调整，考虑检测精度可能下降")
    
    # 4. 提出综合判定规则
    print("\n4. 综合判定规则建议:")
    print("   a. 主要判定标准（满足任一条件）:")
    print("      - 13号、18号或21号染色体Z值绝对值>3")
    print("      - X染色体Z值绝对值>3且伴有其他临床指征")
    print("      - 检测报告明确标记染色体异常")
    print("   b. 次要判定标准（满足两个或以上条件）:")
    print("      - 染色体Z值绝对值在2.5-3之间")
    print("      - X染色体浓度异常（偏离正常范围）")
    print("      - 多个染色体Z值接近异常阈值")
    print("   c. 临床建议:")
    print("      - 符合主要判定标准: 建议羊水穿刺或绒毛取样确诊")
    print("      - 符合次要判定标准: 建议密切监测，考虑重复NIPT检测")
    print("      - 正常结果: 常规产检随访")
    
    # 保存统计学分析结果
    stats_results = []
    for chrom, threshold in chrom_z_thresholds.items():
        stats_results.append({
            '指标': chrom,
            '建议异常阈值': f'>{threshold}倍标准差',
            '临床意义': '染色体非整倍体高风险'
        })
        
    # 添加其他重要指标的参考值
    other_indicators = [
        {'指标': '孕妇BMI指标', '建议异常阈值': '>30或<18.5', '临床意义': '可能影响检测准确性'},
        {'指标': 'X染色体浓度', '建议异常阈值': '偏离正常人群均值±2倍标准差', '临床意义': '可能提示性染色体异常'},
        {'指标': '孕周数值', '建议最佳检测时间': '12-22周', '临床意义': '在此期间检测准确性较高'}
    ]
    
    stats_results.extend(other_indicators)
    
    stats_df = pd.DataFrame(stats_results)
    stats_df.to_excel(os.path.join(results_dir, 'T4_statistical_analysis_results.xlsx'))
    
    print("\nT4任务分析完成！结果已保存到results目录。")
    exit()

# 处理特征列中的缺失值（使用中位数填充）
for col in feature_columns:
    if model_data[col].isnull().sum() > 0:
        median_val = model_data[col].median()
        model_data[col] = model_data[col].fillna(median_val)
        print(f'使用中位数填充 {col} 的缺失值: {median_val:.4f}')

# 处理不平衡数据
# 统计正负样本数量
positive_count = model_data['染色体异常'].sum()
negative_count = len(model_data) - positive_count
print(f'模型数据中异常样本数: {positive_count}')
print(f'模型数据中正常样本数: {negative_count}')

# 如果所有样本都是正常样本（没有异常样本），我们需要调整模型策略
if positive_count == 0:
    print("警告: 没有找到染色体异常的女胎样本。")
    print("将创建一个基于正常样本分布的异常检测模型...")
    
    # 划分训练集和测试集（使用所有数据作为训练集）
    X = model_data[feature_columns]
    
    # 直接进行特征重要性分析和规则制定
    print("\n基于正常样本分布的异常检测方法:")
    
    # 计算各特征的均值和标准差
    feature_stats = []
    for feature in feature_columns:
        mean_val = X[feature].mean()
        std_val = X[feature].std()
        feature_stats.append({
            '特征': feature,
            '均值': mean_val,
            '标准差': std_val,
            '异常下限': mean_val - 3*std_val,
            '异常上限': mean_val + 3*std_val
        })
    
    # 保存特征统计结果
    stats_df = pd.DataFrame(feature_stats)
    stats_df.to_excel(os.path.join(results_dir, 'T4_feature_statistics.xlsx'))
    
    # 可视化特征分布
    plt.figure(figsize=(15, 12))
    for i, feature in enumerate(feature_columns, 1):
        plt.subplot(4, 3, i)
        sns.histplot(X[feature], kde=True)
        mean_val = X[feature].mean()
        std_val = X[feature].std()
        plt.axvline(mean_val - 3*std_val, color='r', linestyle='--', label='异常下限')
        plt.axvline(mean_val + 3*std_val, color='r', linestyle='--', label='异常上限')
        plt.title(feature)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'T4_feature_distributions.png'), dpi=300)
    plt.close()
    
    # 提出异常判定规则
    print("\n异常判定规则建议:")
    print("1. 单指标异常规则: 任意一个特征值超出均值±3倍标准差范围，判定为异常")
    print("2. 多指标异常规则: 多个特征值接近异常阈值（如均值±2.5倍标准差），即使单个指标未超出范围，也应判定为高风险")
    print("3. 重点关注指标: 建议重点关注13号、18号、21号染色体的Z值和X染色体相关指标")
    
    print("\nT4任务分析完成！结果已保存到results目录。")
    exit()

# 如果样本不平衡，进行重采样
if positive_count < negative_count:
    # 过采样正样本
    positive_samples = model_data[model_data['染色体异常'] == 1]
    negative_samples = model_data[model_data['染色体异常'] == 0]
    
    # 过采样正样本以匹配负样本数量
    positive_samples_upsampled = resample(positive_samples,
                                         replace=True,  # 有放回采样
                                         n_samples=negative_count,
                                         random_state=42)
    
    # 合并样本
    model_data = pd.concat([negative_samples, positive_samples_upsampled])
    print(f'重采样后样本数: {len(model_data)}')
    print(f'重采样后异常样本数: {model_data["染色体异常"].sum()}')

# 划分训练集和测试集
X = model_data[feature_columns]
y = model_data['染色体异常']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 模型选择与训练
print("开始模型训练...")
# 定义多种分类器进行比较
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# 训练并评估所有分类器
model_results = {}
for name, clf in classifiers.items():
    print(f'训练 {name} 模型...')
    # 超参数优化
    if name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    elif name == 'Logistic Regression':
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs']
        }
    else:  # SVM
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.1]
        }
    
    # 网格搜索
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train_scaled, y_train)
    
    # 最佳模型预测
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test_scaled)
    y_pred_proba = best_clf.predict_proba(X_test_scaled)[:, 1] if hasattr(best_clf, 'predict_proba') else best_clf.decision_function(X_test_scaled)
    
    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # 保存模型结果
    model_results[name] = {
        'accuracy': accuracy,
        'f1_score': report['macro avg']['f1-score'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'best_params': grid_search.best_params_,
        'model': best_clf
    }
    
    # 保存分类报告
    report_df = pd.DataFrame(report).transpose()
    report_df.to_excel(os.path.join(results_dir, f'T4_classification_report_{name}.xlsx'))
    
    # 保存混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
    plt.title(f'{name}模型混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('实际类别')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'T4_confusion_matrix_{name}.png'), dpi=300)
    plt.close()
    
    # 绘制ROC曲线
    if hasattr(best_clf, 'predict_proba'):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (面积 = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title(f'{name}模型ROC曲线')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'T4_roc_curve_{name}.png'), dpi=300)
        plt.close()

# 6. 选择最佳模型并分析特征重要性
print("选择最佳模型...")
# 找到F1分数最高的模型
best_model_name = max(model_results, key=lambda x: model_results[x]['f1_score'])
best_model = model_results[best_model_name]['model']
print(f'最佳模型: {best_model_name}')
print(f'最佳参数: {model_results[best_model_name]["best_params"]}')
print(f'准确率: {model_results[best_model_name]["accuracy"]:.4f}')
print(f'F1分数: {model_results[best_model_name]["f1_score"]:.4f}')

# 分析特征重要性
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 保存特征重要性
    feature_importance_df = pd.DataFrame({
        '特征': [feature_columns[i] for i in indices],
        '重要性': importances[indices]
    })
    feature_importance_df.to_excel(os.path.join(results_dir, f'T4_feature_importance_{best_model_name}.xlsx'))
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    plt.title(f'{best_model_name}特征重要性')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_columns[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'T4_feature_importance_{best_model_name}.png'), dpi=300)
    plt.close()
    
    print("特征重要性排序:")
    for i, idx in enumerate(indices):
        print(f'{i+1}. {feature_columns[idx]}: {importances[idx]:.4f}')

# 7. 建立女胎异常判定方法
print("建立女胎异常判定方法...")
# 基于最佳模型和特征重要性，提出判定方法

# 保存模型评估结果
model_evaluation_df = pd.DataFrame([
    {
        '模型': name,
        '准确率': results['accuracy'],
        '精确率': results['precision'],
        '召回率': results['recall'],
        'F1分数': results['f1_score']
    } for name, results in model_results.items()
])
model_evaluation_df.to_excel(os.path.join(results_dir, 'T4_model_evaluation_results.xlsx'))

# 8. 总结女胎异常判定方法
print("\n女胎异常判定方法总结:")
print("1. 基于特征重要性分析，以下因素对女胎异常判定最为重要：")
if hasattr(best_model, 'feature_importances_'):
    for i, idx in enumerate(indices[:5]):  # 前5个最重要的特征
        print(f"   {i+1}. {feature_columns[idx]} (重要性: {importances[idx]:.4f})")

print("\n2. 判定流程建议：")
print("   - 首先收集孕妇的BMI、年龄、孕周等基本信息")
print("   - 重点关注13号、18号、21号染色体的Z值和X染色体相关指标")
print("   - 使用优化后的随机森林模型进行综合判定")
print("   - 对于高风险样本，建议结合临床其他检查结果进行进一步确认")

print("\nT4任务分析完成！结果已保存到results目录。")