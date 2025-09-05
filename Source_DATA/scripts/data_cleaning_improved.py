#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
男性胎儿数据清洗脚本 - 改进版
基于T1建模意见进行数据清洗，增强功能和稳定性
"""

import pandas as pd
import numpy as np
import warnings
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import re

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class DataCleaningConfig:
    """数据清洗配置类"""
    # 文件路径配置
    data_files: List[str] = None
    output_dir: str = "cleaned_data"
    
    # 异常值检测阈值
    gestational_age_range: Tuple[float, float] = (10.0, 25.0)
    bmi_range: Tuple[float, float] = (15.0, 60.0)
    y_chromosome_range: Tuple[float, float] = (0.0, 30.0)
    total_reads_min: int = 1000000
    gc_content_range: Tuple[float, float] = (0.3, 0.7)
    
    # 缺失值处理策略
    missing_value_strategy: str = "knn"  # "median", "mean", "knn", "drop"
    knn_neighbors: int = 5
    
    # 质量控制参数
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    zscore_threshold: float = 3.0
    
    # 日志配置
    log_level: str = "INFO"
    save_plots: bool = True

class DataCleaner:
    """数据清洗器类"""
    
    def __init__(self, config: DataCleaningConfig = None):
        self.config = config or DataCleaningConfig()
        self.setup_logging()
        self.column_mapping = {}
        self.cleaning_log = []
        
    def setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_cleaning.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """改进的数据加载功能"""
        self.logger.info("开始加载数据文件...")
        
        # 支持的文件格式
        supported_formats = ['.xlsx', '.xls', '.csv', '.tsv', '.json']
        
        # 如果配置了特定文件，优先使用
        if self.config.data_files:
            for file_path in self.config.data_files:
                if os.path.exists(file_path):
                    return self._load_single_file(file_path)
        
        # 自动搜索数据文件
        current_dir = Path('.')
        for file_path in current_dir.glob('*'):
            if file_path.suffix.lower() in supported_formats:
                self.logger.info(f"发现数据文件: {file_path}")
                df = self._load_single_file(str(file_path))
                if df is not None:
                    return df
        
        self.logger.error("未找到任何支持的数据文件")
        return None
    
    def _load_single_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """加载单个数据文件"""
        try:
            file_path = Path(file_path)
            suffix = file_path.suffix.lower()
            
            if suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif suffix == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            elif suffix == '.tsv':
                df = pd.read_csv(file_path, sep='\t', encoding='utf-8-sig')
            elif suffix == '.json':
                df = pd.read_json(file_path)
            else:
                return None
            
            self.logger.info(f"成功读取文件 {file_path}，数据形状: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"读取文件 {file_path} 失败: {e}")
            return None
    
    def explore_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """增强的数据探索功能"""
        self.logger.info("开始数据探索...")
        
        print("\n=== 数据探索 ===")
        print(f"数据形状: {df.shape}")
        print(f"内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # 基本信息
        print(f"\n列名 ({len(df.columns)}个):")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        # 数据类型分析
        print("\n数据类型分布:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count}列")
        
        # 缺失值分析
        print("\n缺失值统计:")
        missing_stats = df.isnull().sum()
        missing_percent = (missing_stats / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            '缺失数量': missing_stats,
            '缺失比例(%)': missing_percent
        })
        print(missing_df[missing_df['缺失数量'] > 0])
        
        # 数值列统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n数值列统计信息 ({len(numeric_cols)}列):")
            print(df[numeric_cols].describe())
        
        # 分类列分析
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"\n分类列分析 ({len(categorical_cols)}列):")
            for col in categorical_cols[:5]:  # 只显示前5列
                unique_count = df[col].nunique()
                print(f"  {col}: {unique_count}个唯一值")
                if unique_count <= 10:
                    print(f"    值分布: {df[col].value_counts().to_dict()}")
        
        return df
    
    def identify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """增强的列名识别功能"""
        self.logger.info("开始识别关键列...")
        
        print("\n=== 列名识别 ===")
        
        # 预定义的列名模式
        column_patterns = {
            'fetal_sex': [r'胎儿性别', r'fetal.*sex', r'性别'],
            'y_chromosome': [r'Y染色体浓度', r'y.*chromosome', r'Y.*浓度', r'Y.*fraction'],
            'y_z_score': [r'Y染色体的Z值', r'y.*z.*score', r'Y.*Z.*值'],
            'gestational_age': [r'检测孕周', r'gestational.*age', r'孕周', r'GA'],
            'bmi': [r'孕妇BMI', r'bmi', r'BMI', r'体重指数'],
            'height': [r'身高', r'height', r'身长'],
            'weight': [r'体重', r'weight', r'重量'],
            'age': [r'年龄', r'age', r'孕妇年龄'],
            'fetal_health': [r'胎儿是否健康', r'fetal.*health', r'健康状态'],
            'total_reads': [r'原始读段数', r'total.*reads', r'读段数', r'reads'],
            'gc_content': [r'GC含量', r'gc.*content', r'GC.*content']
        }
        
        column_mapping = {}
        
        for key, patterns in column_patterns.items():
            found_col = None
            for pattern in patterns:
                for col in df.columns:
                    if re.search(pattern, col, re.IGNORECASE):
                        found_col = col
                        break
                if found_col:
                    break
            
            if found_col:
                column_mapping[key] = found_col
                print(f"{key}: {found_col} ✓")
            else:
                column_mapping[key] = None
                print(f"{key}: 未找到匹配列 ✗")
        
        # 保存列名映射
        self.column_mapping = column_mapping
        return column_mapping
    
    def detect_outliers(self, df: pd.DataFrame, column: str, method: str = "iqr") -> pd.Series:
        """检测异常值"""
        if column not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        
        data = df[column].dropna()
        if len(data) == 0:
            return pd.Series([False] * len(df), index=df.index)
        
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(data))
            outliers = pd.Series([False] * len(df), index=df.index)
            outliers[data.index] = z_scores > self.config.zscore_threshold
        
        return outliers.fillna(False)
    
    def impute_missing_values(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """智能填充缺失值"""
        if column not in df.columns or df[column].isnull().sum() == 0:
            return df
        
        self.logger.info(f"开始填充 {column} 的缺失值...")
        
        if self.config.missing_value_strategy == "knn":
            # 使用KNN填充
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
                df_filled = df.copy()
                df_filled[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                return df_filled
        
        elif self.config.missing_value_strategy == "median":
            df[column].fillna(df[column].median(), inplace=True)
        
        elif self.config.missing_value_strategy == "mean":
            df[column].fillna(df[column].mean(), inplace=True)
        
        return df
    
    def clean_male_fetal_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """改进的男性胎儿数据清洗"""
        self.logger.info("开始清洗男性胎儿数据...")
        
        print("\n=== 男性胎儿数据清洗 ===")
        df_male = df.copy()
        original_count = len(df_male)
        
        # 1. 筛选男性胎儿样本
        print("1. 筛选男性胎儿样本...")
        y_col = self.column_mapping.get('y_chromosome')
        y_z_col = self.column_mapping.get('y_z_score')
        
        if y_col and y_col in df_male.columns:
            male_mask = df_male[y_col].notna()
            print(f"Y染色体浓度非空样本数: {male_mask.sum()}")
        elif y_z_col and y_z_col in df_male.columns:
            male_mask = df_male[y_z_col].notna()
            print(f"Y染色体Z值非空样本数: {male_mask.sum()}")
        else:
            self.logger.error("未找到Y染色体相关列，无法筛选男性胎儿")
            return None
        
        df_male = df_male[male_mask].copy()
        print(f"筛选后男性胎儿样本数: {len(df_male)}")
        
        # 2. 处理孕周数据
        ga_col = self.column_mapping.get('gestational_age')
        if ga_col and ga_col in df_male.columns:
            print("\n2. 处理孕周数据...")
            df_male = self._process_gestational_age(df_male, ga_col)
        
        # 3. 处理BMI数据
        bmi_col = self.column_mapping.get('bmi')
        if bmi_col and bmi_col in df_male.columns:
            print("\n3. 处理BMI数据...")
            df_male = self._process_bmi_data(df_male, bmi_col)
        
        # 4. 异常值检测和剔除
        print("\n4. 异常值检测和剔除...")
        df_male = self._remove_outliers(df_male)
        
        # 5. 数据质量控制
        print("\n5. 数据质量控制...")
        df_male = self._quality_control(df_male)
        
        # 6. 数据标准化
        print("\n6. 数据标准化...")
        df_male = self._standardize_data(df_male)
        
        # 记录清洗过程
        self.cleaning_log.append({
            'step': 'male_fetal_cleaning',
            'original_count': original_count,
            'cleaned_count': len(df_male),
            'removed_count': original_count - len(df_male)
        })
        
        print(f"\n数据清洗完成！")
        print(f"原始样本数: {original_count}")
        print(f"清洗后样本数: {len(df_male)}")
        print(f"保留比例: {len(df_male)/original_count*100:.1f}%")
        
        return df_male
    
    def _process_gestational_age(self, df: pd.DataFrame, ga_col: str) -> pd.DataFrame:
        """处理孕周数据"""
        # 转换孕周为数值类型
        def parse_gestational_age(ga_str):
            """解析孕周字符串"""
            try:
                if pd.isna(ga_str) or ga_str == '':
                    return np.nan
                ga_str = str(ga_str).strip()
                if 'w' in ga_str:
                    parts = ga_str.split('w')
                    weeks = int(parts[0])
                    if '+' in parts[1]:
                        days = int(parts[1].split('+')[1])
                    else:
                        days = 0
                    return weeks + days / 7.0
                else:
                    return float(ga_str)
            except:
                return np.nan
        
        df[ga_col] = df[ga_col].apply(parse_gestational_age)
        
        # 填充缺失值
        if df[ga_col].isnull().sum() > 0:
            df = self.impute_missing_values(df, ga_col)
        
        return df
    
    def _process_bmi_data(self, df: pd.DataFrame, bmi_col: str) -> pd.DataFrame:
        """处理BMI数据"""
        # 尝试从身高体重计算BMI
        height_col = self.column_mapping.get('height')
        weight_col = self.column_mapping.get('weight')
        
        if (height_col and weight_col and 
            height_col in df.columns and weight_col in df.columns):
            
            # 计算BMI
            height_m = df[height_col] / 100
            calculated_bmi = df[weight_col] / (height_m ** 2)
            
            # 用计算值填充缺失的BMI
            missing_mask = df[bmi_col].isnull()
            df.loc[missing_mask, bmi_col] = calculated_bmi[missing_mask]
            
            print(f"从身高体重计算BMI填充了 {missing_mask.sum()} 个缺失值")
        
        # 填充剩余的缺失值
        if df[bmi_col].isnull().sum() > 0:
            df = self.impute_missing_values(df, bmi_col)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """移除异常值"""
        original_count = len(df)
        
        # 孕周异常值
        ga_col = self.column_mapping.get('gestational_age')
        if ga_col and ga_col in df.columns:
            ga_outliers = ((df[ga_col] < self.config.gestational_age_range[0]) | 
                          (df[ga_col] > self.config.gestational_age_range[1]))
            print(f"孕周异常值数量: {ga_outliers.sum()}")
            df = df[~ga_outliers]
        
        # BMI异常值
        bmi_col = self.column_mapping.get('bmi')
        if bmi_col and bmi_col in df.columns:
            bmi_outliers = ((df[bmi_col] < self.config.bmi_range[0]) | 
                           (df[bmi_col] > self.config.bmi_range[1]))
            print(f"BMI异常值数量: {bmi_outliers.sum()}")
            df = df[~bmi_outliers]
        
        # Y染色体浓度异常值
        y_col = self.column_mapping.get('y_chromosome')
        if y_col and y_col in df.columns:
            y_outliers = ((df[y_col] < self.config.y_chromosome_range[0]) | 
                         (df[y_col] > self.config.y_chromosome_range[1]))
            print(f"Y染色体浓度异常值数量: {y_outliers.sum()}")
            df = df[~y_outliers]
        
        # 使用统计方法检测其他异常值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in [ga_col, bmi_col, y_col]:  # 跳过已处理的列
                continue
            
            outliers = self.detect_outliers(df, col, self.config.outlier_method)
            if outliers.sum() > 0:
                print(f"{col} 异常值数量: {outliers.sum()}")
                df = df[~outliers]
        
        print(f"异常值剔除: {original_count - len(df)} 个样本")
        return df
    
    def _quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据质量控制"""
        original_count = len(df)
        
        # 测序质量控制
        total_reads_col = self.column_mapping.get('total_reads')
        if total_reads_col and total_reads_col in df.columns:
            low_reads = df[total_reads_col] < self.config.total_reads_min
            print(f"低测序深度样本数量: {low_reads.sum()}")
            df = df[~low_reads]
        
        # GC含量质量控制
        gc_col = self.column_mapping.get('gc_content')
        if gc_col and gc_col in df.columns:
            gc_outliers = ((df[gc_col] < self.config.gc_content_range[0]) | 
                          (df[gc_col] > self.config.gc_content_range[1]))
            print(f"GC含量异常样本数量: {gc_outliers.sum()}")
            df = df[~gc_outliers]
        
        print(f"质量控制剔除: {original_count - len(df)} 个样本")
        return df
    
    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据标准化"""
        # Y染色体浓度标准化
        y_col = self.column_mapping.get('y_chromosome')
        if y_col and y_col in df.columns:
            if df[y_col].max() <= 1:
                df[y_col] = df[y_col] * 100
                print("Y染色体浓度已转换为百分比格式")
        
        return df
    
    def generate_visualizations(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame):
        """生成数据可视化"""
        if not self.config.save_plots:
            return
        
        self.logger.info("生成数据可视化...")
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 关键变量分布对比
        key_vars = ['gestational_age', 'bmi', 'y_chromosome']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('数据清洗前后关键变量分布对比', fontsize=16)
        
        for i, var in enumerate(key_vars):
            col = self.column_mapping.get(var)
            if col and col in df_original.columns:
                # 原始数据
                axes[0, i].hist(df_original[col].dropna(), bins=30, alpha=0.7, color='red', label='原始数据')
                axes[0, i].set_title(f'{var} - 原始数据')
                axes[0, i].legend()
                
                # 清洗后数据
                if col in df_cleaned.columns:
                    axes[1, i].hist(df_cleaned[col].dropna(), bins=30, alpha=0.7, color='blue', label='清洗后数据')
                    axes[1, i].set_title(f'{var} - 清洗后数据')
                    axes[1, i].legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.config.output_dir}/data_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 缺失值热图
        if len(df_original) > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df_original.isnull(), cbar=True, yticklabels=False)
            plt.title('原始数据缺失值分布')
            plt.tight_layout()
            plt.savefig(f'{self.config.output_dir}/missing_values_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_summary_report(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> Dict[str, Any]:
        """生成详细的数据清洗报告"""
        self.logger.info("生成数据清洗报告...")
        
        print("\n=== 数据清洗报告 ===")
        
        report = {
            '基本信息': {
                '原始样本数': len(df_original),
                '清洗后样本数': len(df_cleaned),
                '保留比例': f"{len(df_cleaned)/len(df_original)*100:.1f}%",
                '剔除样本数': len(df_original) - len(df_cleaned)
            }
        }
        
        # 核心变量统计
        key_vars = ['gestational_age', 'bmi', 'y_chromosome']
        for var in key_vars:
            col = self.column_mapping.get(var)
            if col and col in df_cleaned.columns:
                stats_dict = {
                    '范围': f"{df_cleaned[col].min():.2f} - {df_cleaned[col].max():.2f}",
                    '中位数': f"{df_cleaned[col].median():.2f}",
                    '均值': f"{df_cleaned[col].mean():.2f}",
                    '标准差': f"{df_cleaned[col].std():.2f}"
                }
                report[f'{var}_统计'] = stats_dict
        
        # 数据质量指标
        quality_metrics = {
            '缺失值比例': f"{(df_cleaned.isnull().sum().sum() / (len(df_cleaned) * len(df_cleaned.columns)) * 100):.2f}%",
            '重复行数': df_cleaned.duplicated().sum(),
            '数据类型一致性': '良好' if df_cleaned.dtypes.nunique() <= 3 else '需要检查'
        }
        report['数据质量指标'] = quality_metrics
        
        # 打印报告
        for section, content in report.items():
            print(f"\n{section}:")
            if isinstance(content, dict):
                for key, value in content.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {content}")
        
        return report
    
    def save_cleaned_data(self, df_cleaned: pd.DataFrame):
        """保存清洗后的数据"""
        self.logger.info("保存清洗后的数据...")
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 保存为多种格式
        output_files = {
            'excel': f'{self.config.output_dir}/male_fetal_data_cleaned.xlsx',
            'csv': f'{self.config.output_dir}/male_fetal_data_cleaned.csv',
            'json': f'{self.config.output_dir}/male_fetal_data_cleaned.json'
        }
        
        for format_name, file_path in output_files.items():
            try:
                if format_name == 'excel':
                    df_cleaned.to_excel(file_path, index=False)
                elif format_name == 'csv':
                    df_cleaned.to_csv(file_path, index=False, encoding='utf-8-sig')
                elif format_name == 'json':
                    df_cleaned.to_json(file_path, orient='records', force_ascii=False, indent=2)
                
                print(f"已保存清洗后的数据到: {file_path}")
            except Exception as e:
                self.logger.error(f"保存{format_name}文件失败: {e}")
        
        # 保存配置和日志
        self._save_metadata(df_cleaned)
    
    def _save_metadata(self, df_cleaned: pd.DataFrame):
        """保存元数据"""
        metadata = {
            'column_mapping': self.column_mapping,
            'cleaning_log': self.cleaning_log,
            'config': self.config.__dict__,
            'data_info': {
                'shape': df_cleaned.shape,
                'columns': list(df_cleaned.columns),
                'dtypes': df_cleaned.dtypes.to_dict()
            }
        }
        
        with open(f'{self.config.output_dir}/cleaning_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"已保存元数据到: {self.config.output_dir}/cleaning_metadata.json")

def main():
    """主函数"""
    print("=== 男性胎儿数据清洗程序 - 改进版 ===")
    
    # 创建配置
    config = DataCleaningConfig(
        data_files=['data.xlsx', 'dataA.csv', 'dataB.csv'],
        output_dir="cleaned_data",
        missing_value_strategy="knn",
        outlier_method="iqr",
        save_plots=True
    )
    
    # 创建数据清洗器
    cleaner = DataCleaner(config)
    
    try:
        # 1. 加载数据
        df = cleaner.load_data()
        if df is None:
            print("数据加载失败，程序退出")
            return
        
        # 2. 探索数据
        df = cleaner.explore_data(df)
        
        # 3. 识别关键列
        column_mapping = cleaner.identify_columns(df)
        
        # 4. 清洗男性胎儿数据
        df_cleaned = cleaner.clean_male_fetal_data(df)
        if df_cleaned is None:
            print("数据清洗失败，程序退出")
            return
        
        # 5. 生成可视化
        cleaner.generate_visualizations(df, df_cleaned)
        
        # 6. 生成报告
        report = cleaner.generate_summary_report(df, df_cleaned)
        
        # 7. 保存清洗后的数据
        cleaner.save_cleaned_data(df_cleaned)
        
        print("\n=== 数据清洗完成 ===")
        
    except Exception as e:
        cleaner.logger.error(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
