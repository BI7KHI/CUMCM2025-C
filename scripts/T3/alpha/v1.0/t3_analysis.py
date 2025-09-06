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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils import resample
import os

# 设置字体显示 - 使用系统可用字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# 对于中文字符，我们将使用英文标签以确保兼容性

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 脚本在 scripts/T3/v1.0/ 中，所以需要三级目录回到项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))

# 创建结果目录
results_dir = os.path.join(project_root, 'results', 'T3', 'alpha', 'v1.0')
os.makedirs(results_dir, exist_ok=True)

# 读取数据（不使用列名）
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

# 数据预处理：转换数值型列的数据类型
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
    data[col] = data[col].apply(safe_float_convert)

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

data['孕周数值'] = data['孕妇本次检测时的孕周'].apply(convert_gestational_age)

# 任务T3：分析X染色体浓度异常对NIPT检测结果的影响

# 1. 区分胎儿性别
# 男胎：Y染色体浓度非空且>0
male_fetus_data = data[(data['Y染色体浓度'].notna()) & (data['Y染色体浓度'] > 0)].copy()
# 女胎：Y染色体浓度为0或空
female_fetus_data = data[(data['Y染色体浓度'].isna()) | (data['Y染色体浓度'] == 0)].copy()

print(f'男胎样本数: {len(male_fetus_data)}')
print(f'女胎样本数: {len(female_fetus_data)}')

# 2. 分析X染色体浓度的分布特性
plt.figure(figsize=(12, 6))

# 男胎X染色体浓度分布
sns.kdeplot(male_fetus_data['X染色体浓度'].dropna(), label='Male Fetus', fill=True, alpha=0.5)
# 女胎X染色体浓度分布
sns.kdeplot(female_fetus_data['X染色体浓度'].dropna(), label='Female Fetus', fill=True, alpha=0.5)

plt.title('X Chromosome Concentration Distribution (Male vs Female Fetus)')
plt.xlabel('X Chromosome Concentration')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T3_alpha_v1.0.0.png'), dpi=300)
plt.close()

# 3. 确定X染色体浓度的正常范围
# 对于男胎，正常情况下X染色体浓度应该在某个范围内
male_x_concentration = male_fetus_data['X染色体浓度'].dropna()
female_x_concentration = female_fetus_data['X染色体浓度'].dropna()

# 使用统计学方法确定正常范围（例如，均值±2标准差）
male_mean = male_x_concentration.mean()
male_std = male_x_concentration.std()
male_lower_bound = male_mean - 2 * male_std
male_upper_bound = male_mean + 2 * male_std

female_mean = female_x_concentration.mean()
female_std = female_x_concentration.std()
female_lower_bound = female_mean - 2 * female_std
female_upper_bound = female_mean + 2 * female_std

print(f'男胎X染色体浓度正常范围: {male_lower_bound:.4f} - {male_upper_bound:.4f}')
print(f'女胎X染色体浓度正常范围: {female_lower_bound:.4f} - {female_upper_bound:.4f}')

# 4. 分析X染色体浓度异常与染色体异常的关系
# 标记X染色体浓度异常的样本
male_fetus_data['X染色体浓度异常'] = ((male_fetus_data['X染色体浓度'] < male_lower_bound) | \
                                   (male_fetus_data['X染色体浓度'] > male_upper_bound))

female_fetus_data['X染色体浓度异常'] = ((female_fetus_data['X染色体浓度'] < female_lower_bound) | \
                                     (female_fetus_data['X染色体浓度'] > female_upper_bound))

# 合并数据进行分析
combined_data = pd.concat([male_fetus_data, female_fetus_data])

# 分析染色体异常情况
def analyze_chromosome_abnormalities(df):
    # 染色体异常列的处理
    # 过滤掉空值
    abnormal_data = df[df['检测出的染色体异常'].notna()]
    # 提取异常类型
    abnormal_types = abnormal_data['检测出的染色体异常'].value_counts()
    print("染色体异常类型统计:")
    print(abnormal_types)
    
    # 计算X染色体浓度异常样本中染色体异常的比例
    x_abnormal_df = df[df['X染色体浓度异常'] == True]
    if len(x_abnormal_df) > 0:
        x_abnormal_chromosome_ratio = x_abnormal_df['检测出的染色体异常'].notna().mean()
        print(f'X染色体浓度异常样本中染色体异常的比例: {x_abnormal_chromosome_ratio:.2%}')
    
    # 计算X染色体浓度正常样本中染色体异常的比例
    x_normal_df = df[df['X染色体浓度异常'] == False]
    if len(x_normal_df) > 0:
        x_normal_chromosome_ratio = x_normal_df['检测出的染色体异常'].notna().mean()
        print(f'X染色体浓度正常样本中染色体异常的比例: {x_normal_chromosome_ratio:.2%}')
    
    # 卡方检验
    if len(x_abnormal_df) > 0 and len(x_normal_df) > 0:
        # 构建列联表
        contingency_table = [
            [len(x_abnormal_df[x_abnormal_df['检测出的染色体异常'].notna()]), 
             len(x_abnormal_df[x_abnormal_df['检测出的染色体异常'].isna()])],
            [len(x_normal_df[x_normal_df['检测出的染色体异常'].notna()]), 
             len(x_normal_df[x_normal_df['检测出的染色体异常'].isna()])]
        ]
        
        # 卡方检验
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        print(f'卡方检验结果: chi2={chi2:.4f}, p-value={p_value:.4f}')
        if p_value < 0.05:
            print("X染色体浓度异常与染色体异常存在显著关联。")
        else:
            print("X染色体浓度异常与染色体异常不存在显著关联。")

analyze_chromosome_abnormalities(combined_data)

# 由于女胎样本数极少，我们在建立模型时只使用男胎数据进行分析
model_data = male_fetus_data.dropna(subset=['检测出的染色体异常', 'X染色体浓度']).copy()
model_data['标签'] = model_data['检测出的染色体异常'].notna().astype(int)

# 5. 建立数学模型用于分类判断
# 准备特征和标签
# 定义特征列
feature_columns = ['孕妇年龄', '孕妇BMI指标', '孕周数值', 'X染色体浓度', 'Y染色体浓度',
                   'GC含量', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',
                   'X染色体的Z值', 'Y染色体的Z值', '13号染色体的GC含量', '18号染色体的GC含量',
                   '21号染色体的GC含量']

# 处理不平衡数据
# 统计正负样本数量
positive_count = model_data['标签'].sum()
negative_count = len(model_data) - positive_count
print(f'正样本数（染色体异常）: {positive_count}')
print(f'负样本数（染色体正常）: {negative_count}')

# 如果所有样本都是正样本，我们需要改变模型策略
if negative_count == 0:
    print("警告: 所有样本都被标记为染色体异常。我们需要调整模型策略。")
    # 在这种情况下，我们可以改为预测X染色体浓度是否异常
    # 重新定义标签为X染色体浓度异常
    model_data['标签'] = model_data['X染色体浓度异常'].astype(int)
    # 重新统计正负样本
    positive_count = model_data['标签'].sum()
    negative_count = len(model_data) - positive_count
    print(f'重新定义标签后，正样本数（X染色体浓度异常）: {positive_count}')
    print(f'重新定义标签后，负样本数（X染色体浓度正常）: {negative_count}')
    
    # 如果仍然没有负样本，我们无法建立二分类模型
    if negative_count == 0:
        print("错误: 无法建立分类模型，因为所有样本都是同一类别。")
        exit()

# 如果数据不平衡，进行重采样
balanced_data = model_data.copy()
if positive_count < negative_count * 0.2 or positive_count > negative_count * 5:
    # 分离正负样本
    positive_samples = model_data[model_data['标签'] == 1]
    negative_samples = model_data[model_data['标签'] == 0]
    
    # 确定重采样大小
    sample_size = min(positive_count, negative_count)
    
    # 对多数类进行下采样
    if negative_count > positive_count:
        negative_downsampled = resample(negative_samples, replace=False, n_samples=sample_size, random_state=42)
        balanced_data = pd.concat([positive_samples, negative_downsampled])
    else:
        # 对少数类进行上采样
        positive_upsampled = resample(positive_samples, replace=True, n_samples=sample_size, random_state=42)
        balanced_data = pd.concat([positive_upsampled, negative_samples])

# 划分训练集和测试集
X = balanced_data[feature_columns]
y = balanced_data['标签']

# 处理特征中的缺失值
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 尝试多种分类模型并选择最佳模型
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

# 评估指标存储
evaluation_results = {}

# 训练和评估每个模型
for name, model in models.items():
    print(f'\n训练模型: {name}')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 计算评估指标
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # 计算准确率、精确率、召回率、F1分数
    accuracy = report['accuracy']
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1_score = report['1']['f1-score']
    
    # 存储结果
    evaluation_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': cm,
        'model': model
    }
    
    print(f'准确率: {accuracy:.4f}')
    print(f'精确率: {precision:.4f}')
    print(f'召回率: {recall:.4f}')
    print(f'F1分数: {f1_score:.4f}')
    print('混淆矩阵:')
    print(cm)

# 7. 选择最佳模型并进行优化
# 根据F1分数选择最佳模型
best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['f1_score'])
best_model = evaluation_results[best_model_name]['model']

print(f'\n最佳模型: {best_model_name}')
print(f'F1分数: {evaluation_results[best_model_name]["f1_score"]:.4f}')

# 对最佳模型进行超参数优化
if best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
elif best_model_name == 'SVM':
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1]
    }

# 网格搜索优化
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f'优化后的最佳参数: {grid_search.best_params_}')
print(f'优化后的最佳F1分数: {grid_search.best_score_:.4f}')

# 使用优化后的模型进行预测
best_optimized_model = grid_search.best_estimator_
y_pred_optimized = best_optimized_model.predict(X_test_scaled)
y_pred_proba_optimized = best_optimized_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_optimized_model, 'predict_proba') else None

# 评估优化后的模型
optimized_report = classification_report(y_test, y_pred_optimized)
optimized_cm = confusion_matrix(y_test, y_pred_optimized)

print('\n优化后模型的分类报告:')
print(optimized_report)
print('优化后模型的混淆矩阵:')
print(optimized_cm)

# 8. 模型解释：特征重要性分析
# 创建中英文特征名映射
feature_name_mapping = {
    '孕妇年龄': 'Maternal Age',
    '孕妇BMI指标': 'Maternal BMI',
    '孕周数值': 'Gestational Age',
    'X染色体浓度': 'X Chr Concentration',
    'Y染色体浓度': 'Y Chr Concentration',
    'GC含量': 'GC Content',
    '13号染色体的Z值': 'Chr13 Z-score',
    '18号染色体的Z值': 'Chr18 Z-score',
    '21号染色体的Z值': 'Chr21 Z-score',
    'X染色体的Z值': 'X Chr Z-score',
    'Y染色体的Z值': 'Y Chr Z-score',
    '13号染色体的GC含量': 'Chr13 GC Content',
    '18号染色体的GC含量': 'Chr18 GC Content',
    '21号染色体的GC含量': 'Chr21 GC Content'
}

if hasattr(best_optimized_model, 'feature_importances_'):
    # 获取特征重要性
    feature_importance = best_optimized_model.feature_importances_
    # 创建特征重要性DataFrame，使用英文特征名
    feature_importance_df = pd.DataFrame({
        'Feature': [feature_name_mapping.get(f, f) for f in feature_columns],
        'Importance': feature_importance
    })
    # 按重要性排序
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    print('\n特征重要性排序:')
    for i, row in feature_importance_df.iterrows():
        print(f'{row["Feature"]}: {row["Importance"]:.4f}')
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f'Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'T3_alpha_v1.0.0{best_model_name}.png'), dpi=300)
    plt.close()
elif hasattr(best_optimized_model, 'coef_'):
    # 对于逻辑回归，获取系数
    coefs = best_optimized_model.coef_[0]
    feature_coef_df = pd.DataFrame({
        'Feature': [feature_name_mapping.get(f, f) for f in feature_columns],
        'Coefficient': coefs
    })
    # 按绝对值排序
    feature_coef_df['Abs_Value'] = feature_coef_df['Coefficient'].abs()
    feature_coef_df = feature_coef_df.sort_values('Abs_Value', ascending=False)
    
    print('\n特征系数排序:')
    for i, row in feature_coef_df.iterrows():
        print(f'{row["Feature"]}: {row["Coefficient"]:.4f}')
    
    # 可视化特征系数
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=feature_coef_df)
    plt.title(f'Feature Coefficients - {best_model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'T3_alpha_v1.0.0{best_model_name}.png'), dpi=300)
    plt.close()

# 9. 绘制ROC曲线和PR曲线
def plot_metrics(y_true, y_pred_proba, model_name, save_dir):
    if y_pred_proba is not None:
        # ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'T3_alpha_v1.0.0{model_name}.png'), dpi=300)
        plt.close()
        
        # PR曲线
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'T3_alpha_v1.0.0{model_name}.png'), dpi=300)
        plt.close()
    
# 绘制优化后模型的评估曲线
plot_metrics(y_test, y_pred_proba_optimized, f'{best_model_name}_optimized', results_dir)

# 10. 分析X染色体异常对NIPT检测结果的影响程度
# 计算X染色体浓度异常与其他染色体异常的相关性
# 提取染色体异常相关的列
chromosome_abnormal_columns = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']

# 计算相关性矩阵
correlation_data = combined_data.dropna(subset=chromosome_abnormal_columns + ['X染色体浓度异常'])
correlation_matrix = correlation_data[chromosome_abnormal_columns + ['X染色体浓度异常']].corr()

# 可视化相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation: X Chromosome Concentration Abnormality vs Other Chromosome Abnormalities')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T3_alpha_v1.0.0.png'), dpi=300)
plt.close()

# 11. 分析X染色体浓度异常对检测准确性的影响
# 检测准确性定义为正确识别染色体异常的能力
# 这里用胎儿健康状况作为参考标准
healthy_data = combined_data[combined_data['胎儿是否健康'] == '健康']
unhealthy_data = combined_data[combined_data['胎儿是否健康'] == '不健康']

# 计算X染色体浓度异常在健康和不健康胎儿中的比例
if len(healthy_data) > 0:
    healthy_x_abnormal_ratio = healthy_data['X染色体浓度异常'].mean()
    print(f'健康胎儿中X染色体浓度异常的比例: {healthy_x_abnormal_ratio:.2%}')

if len(unhealthy_data) > 0:
    unhealthy_x_abnormal_ratio = unhealthy_data['X染色体浓度异常'].mean()
    print(f'不健康胎儿中X染色体浓度异常的比例: {unhealthy_x_abnormal_ratio:.2%}')

# 计算检测准确率
# 真阳性：检测出异常且胎儿确实不健康
# 真阴性：检测正常且胎儿健康
# 假阳性：检测出异常但胎儿健康
# 假阴性：检测正常但胎儿不健康

# 准备数据
accuracy_data = combined_data.dropna(subset=['胎儿是否健康'])
accuracy_data['检测结果'] = accuracy_data['检测出的染色体异常'].notna().astype(int)
accuracy_data['实际健康状况'] = (accuracy_data['胎儿是否健康'] == '健康').astype(int)

# 计算混淆矩阵
accuracy_cm = confusion_matrix(accuracy_data['实际健康状况'], accuracy_data['检测结果'])

# 计算准确率、精确率、召回率
if accuracy_cm.size > 0:
    if accuracy_cm.shape[0] > 1 and accuracy_cm.shape[1] > 1:
        tn, fp, fn, tp = accuracy_cm.ravel()
        detection_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        detection_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        detection_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f'\nNIPT检测准确性:')
        print(f'准确率: {detection_accuracy:.4f}')
        print(f'精确率: {detection_precision:.4f}')
        print(f'召回率: {detection_recall:.4f}')
        print('检测混淆矩阵:')
        print(accuracy_cm)

# 12. 保存模型和结果
# 保存评估结果到Excel
results_list = []
for name, metrics in evaluation_results.items():
    results_list.append({
        '模型': name,
        '准确率': metrics['accuracy'],
        '精确率': metrics['precision'],
        '召回率': metrics['recall'],
        'F1分数': metrics['f1_score']
    })

# 添加优化后的模型结果
optimized_report_dict = classification_report(y_test, y_pred_optimized, output_dict=True)
results_list.append({
    '模型': f'{best_model_name}_optimized',
    '准确率': optimized_report_dict['accuracy'],
    '精确率': optimized_report_dict['1']['precision'],
    '召回率': optimized_report_dict['1']['recall'],
    'F1分数': optimized_report_dict['1']['f1-score']
})

results_df = pd.DataFrame(results_list)
results_df.to_excel(os.path.join(results_dir, 'T3_alpha_v1.0.0.xlsx'), index=False)

# 保存最佳模型的特征重要性或系数
if 'feature_importance_df' in locals():
    feature_importance_df.to_excel(os.path.join(results_dir, 'T3_alpha_v1.0.0.xlsx'), index=False)
elif 'feature_coef_df' in locals():
    feature_coef_df.to_excel(os.path.join(results_dir, 'T3_alpha_v1.0.0.xlsx'), index=False)

# 保存相关性矩阵
correlation_matrix.to_excel(os.path.join(results_dir, 'T3_alpha_v1.0.0.xlsx'))

print("\nT3任务分析完成！结果已保存到results目录。")