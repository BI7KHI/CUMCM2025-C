import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# 如果有中文字体问题，可以使用英文标签
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

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
# 列索引: 0-样本序号, 1-孕妇代码, 2-孕妇年龄, 3-孕妇身高, 4-孕妇体重, 5-末次月经时间,
# 6-IVF妊娠方式, 7-检测时间, 8-检测抽血次数, 9-孕妇本次检测时的孕周, 10-孕妇BMI指标,
# 11-原始测序数据的总读段数, 12-总读段数中在参考基因组上比对的比例, 13-总读段数中重复读段的比例,
# 14-总读段数中唯一比对的读段数, 15-GC含量, 16-13号染色体的Z值, 17-18号染色体的Z值,
# 18-21号染色体的Z值, 19-X染色体的Z值, 20-Y染色体的Z值, 21-Y染色体浓度,
# 22-X染色体浓度, 23-13号染色体的GC含量, 24-18号染色体的GC含量, 25-21号染色体的GC含量,
# 26-被过滤掉的读段数占总读段数的比例, 27-检测出的染色体异常, 28-孕妇的怀孕次数,
# 29-孕妇的生产次数, 30-胎儿是否健康

# 数据预处理：只保留男胎数据（即Y染色体浓度非空的行）
male_fetus_data = data[data.iloc[:, 21].notna()].copy()

# 将数据转换为DataFrame，并添加列名以便后续处理
columns = ['样本序号', '孕妇代码', '孕妇年龄', '孕妇身高', '孕妇体重', '末次月经时间',
           'IVF妊娠方式', '检测时间', '检测抽血次数', '孕妇本次检测时的孕周', '孕妇BMI指标',
           '原始测序数据的总读段数', '总读段数中在参考基因组上比对的比例', '总读段数中重复读段的比例',
           '总读段数中唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值',
           '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度',
           'X染色体浓度', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
           '被过滤掉的读段数占总读段数的比例', '检测出的染色体异常', '孕妇的怀孕次数',
           '孕妇的生产次数', '胎儿是否健康']
male_fetus_data.columns = columns

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

# 相关分析：计算Y染色体浓度与各指标的相关系数
correlation_vars = ['孕周数值', '孕妇BMI指标', '孕妇年龄', '孕妇身高', '孕妇体重', 
                   '原始测序数据的总读段数', '总读段数中在参考基因组上比对的比例', '总读段数中重复读段的比例']
correlation_results = {}

for var in correlation_vars:
    # 过滤掉NaN值
    valid_data = male_fetus_data[[var, 'Y染色体浓度']].dropna()
    if len(valid_data) > 1:
        corr_coef, p_value = stats.pearsonr(valid_data['Y染色体浓度'], valid_data[var])
        correlation_results[var] = {'相关系数': corr_coef, 'p值': p_value}
    else:
        correlation_results[var] = {'相关系数': np.nan, 'p值': np.nan}

# 保存相关性分析结果
correlation_df = pd.DataFrame(correlation_results).T
correlation_df.to_excel(os.path.join(results_dir, 'T1_correlation_results.xlsx'))

print("相关性分析结果:")
print(correlation_df)

# 可视化相关性热力图
plt.figure(figsize=(10, 8))
# 选择有效的变量进行相关性分析
valid_vars = ['Y染色体浓度'] + [var for var in correlation_vars if not np.isnan(correlation_results[var]['相关系数'])]
correlation_matrix = male_fetus_data[valid_vars].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Y染色体浓度与各指标的相关性热力图')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T1_correlation_heatmap.png'), dpi=300)
plt.close()

# 多元线性回归模型：以Y染色体浓度为因变量，其他指标为自变量
# 过滤掉包含NaN值的行
regression_data = male_fetus_data[['孕周数值', '孕妇BMI指标', '孕妇年龄', '孕妇身高', '孕妇体重', 'Y染色体浓度']].dropna()

X = regression_data[['孕周数值', '孕妇BMI指标', '孕妇年龄', '孕妇身高', '孕妇体重']]
X = sm.add_constant(X)  # 添加常数项
y = regression_data['Y染色体浓度']

# 建立模型
model = sm.OLS(y, X)
results = model.fit()

# 保存回归结果
with open(os.path.join(results_dir, 'T1_regression_results.txt'), 'w', encoding='utf-8') as f:
    f.write(str(results.summary()))

print("\n回归分析结果:")
print(results.summary())

# 残差分析
plt.figure(figsize=(12, 8))

# 残差图
plt.subplot(2, 2, 1)
plt.scatter(results.predict(), results.resid)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')

# Q-Q图
plt.subplot(2, 2, 2)
stats.probplot(results.resid, dist="norm", plot=plt)
plt.title('Q-Q图')

# 残差直方图
plt.subplot(2, 2, 3)
plt.hist(results.resid, bins=30)
plt.xlabel('残差')
plt.ylabel('频数')
plt.title('残差直方图')

# 标准化残差绝对值与预测值的关系
plt.subplot(2, 2, 4)
plt.scatter(results.predict(), np.abs(results.get_influence().resid_studentized_internal))
plt.xlabel('预测值')
plt.ylabel('标准化残差绝对值')
plt.title('标准化残差绝对值与预测值的关系')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T1_residual_analysis.png'), dpi=300)
plt.close()

# 单独分析Y染色体浓度与孕周和BMI的关系
# 过滤掉包含NaN值的行
scatter_data = male_fetus_data[['孕周数值', '孕妇BMI指标', 'Y染色体浓度']].dropna()

# 1. Y染色体浓度与孕周的散点图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(scatter_data['孕周数值'], scatter_data['Y染色体浓度'], alpha=0.5)
# 添加趋势线
z = np.polyfit(scatter_data['孕周数值'], scatter_data['Y染色体浓度'], 1)
f = np.poly1d(z)
plt.plot(scatter_data['孕周数值'], f(scatter_data['孕周数值']), 'r--')
plt.xlabel('孕周')
plt.ylabel('Y染色体浓度')
plt.title('Y染色体浓度与孕周的关系')

# 2. Y染色体浓度与BMI的散点图
plt.subplot(1, 2, 2)
plt.scatter(scatter_data['孕妇BMI指标'], scatter_data['Y染色体浓度'], alpha=0.5)
# 添加趋势线
z = np.polyfit(scatter_data['孕妇BMI指标'], scatter_data['Y染色体浓度'], 1)
f = np.poly1d(z)
plt.plot(scatter_data['孕妇BMI指标'], f(scatter_data['孕妇BMI指标']), 'r--')
plt.xlabel('BMI')
plt.ylabel('Y染色体浓度')
plt.title('Y染色体浓度与BMI的关系')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'T1_y_chromosome_vs_gestational_age_bmi.png'), dpi=300)
plt.close()

print("\nT1任务分析完成！结果已保存到results目录。")