# 数据清洗脚本改进版

## 主要改进内容

### 1. 架构优化
- **面向对象设计**: 使用 `DataCleaner` 类封装所有功能
- **配置管理**: 通过 `DataCleaningConfig` 类管理所有参数
- **模块化设计**: 每个功能独立成方法，便于维护和扩展

### 2. 数据加载增强
- **多格式支持**: 支持 Excel (.xlsx, .xls)、CSV、TSV、JSON 格式
- **自动文件发现**: 自动搜索当前目录下的数据文件
- **编码处理**: 自动处理 UTF-8-BOM 编码问题
- **错误处理**: 完善的异常处理和日志记录

### 3. 列名识别改进
- **模糊匹配**: 使用正则表达式进行模糊匹配
- **多语言支持**: 支持中英文列名识别
- **自动检测**: 智能识别关键列，减少手动配置

### 4. 异常值检测优化
- **多种方法**: 支持 IQR、Z-Score、Isolation Forest 等方法
- **可视化支持**: 自动生成异常值检测图表
- **统计方法**: 使用科学统计方法而非简单阈值

### 5. 缺失值处理智能化
- **KNN填充**: 使用 KNN 算法进行智能填充
- **多种策略**: 支持中位数、均值、KNN、删除等策略
- **关联填充**: 利用相关变量进行更准确的填充

### 6. 数据质量控制增强
- **多维度检查**: 从多个角度检查数据质量
- **动态阈值**: 根据数据分布动态调整质量控制阈值
- **质量报告**: 生成详细的数据质量报告

### 7. 可视化和报告
- **对比图表**: 清洗前后数据分布对比
- **缺失值热图**: 直观显示缺失值分布
- **详细报告**: 包含统计指标、质量指标等
- **元数据保存**: 保存清洗过程和配置信息

### 8. 日志和配置
- **完整日志**: 记录所有操作和错误信息
- **配置文件**: 支持 JSON 配置文件
- **元数据**: 保存清洗过程的完整元数据

## 使用方法

### 基本使用
```python
from data_cleaning_improved import DataCleaner, DataCleaningConfig

# 使用默认配置
cleaner = DataCleaner()
df_cleaned = cleaner.clean_male_fetal_data(df)
```

### 自定义配置
```python
# 创建自定义配置
config = DataCleaningConfig(
    data_files=['my_data.xlsx'],
    output_dir='my_output',
    missing_value_strategy='knn',
    outlier_method='zscore',
    save_plots=True
)

# 使用自定义配置
cleaner = DataCleaner(config)
```

### 从配置文件加载
```python
import json

# 加载配置文件
with open('config.json', 'r', encoding='utf-8') as f:
    config_dict = json.load(f)

config = DataCleaningConfig(**config_dict['data_cleaning_config'])
cleaner = DataCleaner(config)
```

## 主要功能对比

| 功能 | 原版本 | 改进版本 |
|------|--------|----------|
| 数据加载 | 硬编码文件名 | 多格式支持，自动发现 |
| 列名识别 | 固定映射 | 模糊匹配，自动检测 |
| 异常值检测 | 简单阈值 | 统计方法，多种算法 |
| 缺失值处理 | 中位数填充 | KNN等智能方法 |
| 质量控制 | 基础检查 | 多维度质量控制 |
| 可视化 | 无 | 丰富的图表和报告 |
| 日志记录 | 简单打印 | 完整日志系统 |
| 配置管理 | 硬编码 | 配置文件支持 |
| 错误处理 | 基础 | 完善的异常处理 |

## 输出文件

清洗完成后会生成以下文件：
- `cleaned_data/male_fetal_data_cleaned.xlsx` - Excel格式清洗数据
- `cleaned_data/male_fetal_data_cleaned.csv` - CSV格式清洗数据
- `cleaned_data/male_fetal_data_cleaned.json` - JSON格式清洗数据
- `cleaned_data/data_distribution_comparison.png` - 数据分布对比图
- `cleaned_data/missing_values_heatmap.png` - 缺失值热图
- `cleaned_data/cleaning_metadata.json` - 清洗元数据
- `data_cleaning.log` - 详细日志文件

## 依赖包

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## 注意事项

1. 确保数据文件在正确的位置
2. 检查列名是否符合预期模式
3. 根据实际数据调整配置参数
4. 查看日志文件了解详细处理过程
5. 检查输出图表验证清洗效果
