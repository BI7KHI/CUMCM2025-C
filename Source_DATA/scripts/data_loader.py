#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载工具
解决数据文件加载问题
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

def load_data_robust() -> Optional[pd.DataFrame]:
    """
    健壮的数据加载函数
    尝试多种方式加载数据文件
    """
    print("正在尝试加载数据文件...")
    
    # 定义可能的文件路径
    possible_files = [
        'data.xlsx',
        'dataA.csv', 
        'dataB.csv',
        'male_fetal_data_cleaned.xlsx',
        'male_fetal_data_cleaned.csv'
    ]
    
    # 尝试加载每个文件
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"发现文件: {file_path}")
            try:
                df = load_single_file(file_path)
                if df is not None:
                    print(f"成功加载文件: {file_path}")
                    print(f"数据形状: {df.shape}")
                    return df
            except Exception as e:
                print(f"加载文件 {file_path} 失败: {e}")
                continue
    
    print("未找到可加载的数据文件")
    return None

def load_single_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    加载单个数据文件
    """
    try:
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix in ['.xlsx', '.xls']:
            # 尝试不同的Excel加载方式
            try:
                df = pd.read_excel(file_path)
            except Exception as e1:
                print(f"标准Excel加载失败: {e1}")
                try:
                    # 尝试指定引擎
                    df = pd.read_excel(file_path, engine='openpyxl')
                except Exception as e2:
                    print(f"openpyxl引擎加载失败: {e2}")
                    try:
                        df = pd.read_excel(file_path, engine='xlrd')
                    except Exception as e3:
                        print(f"xlrd引擎加载失败: {e3}")
                        raise e3
        
        elif suffix == '.csv':
            # 尝试不同的CSV加载方式
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='gbk')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8-sig')
                    except Exception as e:
                        print(f"CSV编码尝试失败: {e}")
                        raise e
        
        else:
            print(f"不支持的文件格式: {suffix}")
            return None
        
        # 检查数据是否为空
        if df.empty:
            print(f"文件 {file_path} 为空")
            return None
        
        print(f"成功加载 {file_path}: {df.shape[0]} 行, {df.shape[1]} 列")
        return df
        
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def identify_columns_robust(df: pd.DataFrame) -> Dict[str, str]:
    """
    健壮的列名识别函数
    """
    print("正在识别关键列...")
    
    # 列名映射模式
    column_patterns = {
        'bmi': [r'BMI', r'bmi', r'体重指数', r'孕妇BMI'],
        'y_chromosome': [r'Y染色体浓度', r'Y.*浓度', r'y.*chromosome', r'Y.*fraction'],
        'y_z_score': [r'Y染色体的Z值', r'Y.*Z.*值', r'y.*z.*score'],
        'gestational_age': [r'检测孕周', r'孕周', r'gestational.*age', r'GA'],
        'age': [r'年龄', r'age'],
        'height': [r'身高', r'height'],
        'weight': [r'体重', r'weight'],
        'fetal_health': [r'胎儿是否健康', r'fetal.*health', r'健康状态']
    }
    
    column_mapping = {}
    
    for key, patterns in column_patterns.items():
        found_col = None
        for pattern in patterns:
            for col in df.columns:
                if pattern.lower() in col.lower():
                    found_col = col
                    break
            if found_col:
                break
        
        if found_col:
            column_mapping[key] = found_col
            print(f"  {key}: {found_col} ✓")
        else:
            column_mapping[key] = None
            print(f"  {key}: 未找到 ✗")
    
    return column_mapping

def validate_data(df: pd.DataFrame, column_mapping: Dict[str, str]) -> bool:
    """
    验证数据是否满足分析要求
    """
    print("正在验证数据...")
    
    required_columns = ['bmi', 'y_chromosome', 'gestational_age']
    missing_columns = []
    
    for col_key in required_columns:
        if col_key not in column_mapping or column_mapping[col_key] is None:
            missing_columns.append(col_key)
    
    if missing_columns:
        print(f"缺少必要的列: {missing_columns}")
        return False
    
    # 检查数据质量
    bmi_col = column_mapping['bmi']
    y_col = column_mapping['y_chromosome']
    ga_col = column_mapping['gestational_age']
    
    # 检查缺失值
    missing_bmi = df[bmi_col].isnull().sum()
    missing_y = df[y_col].isnull().sum()
    missing_ga = df[ga_col].isnull().sum()
    
    print(f"缺失值统计:")
    print(f"  BMI: {missing_bmi} ({missing_bmi/len(df)*100:.1f}%)")
    print(f"  Y染色体浓度: {missing_y} ({missing_y/len(df)*100:.1f}%)")
    print(f"  孕周: {missing_ga} ({missing_ga/len(df)*100:.1f}%)")
    
    # 检查数据范围
    print(f"数据范围:")
    print(f"  BMI: {df[bmi_col].min():.2f} - {df[bmi_col].max():.2f}")
    print(f"  Y染色体浓度: {df[y_col].min():.2f} - {df[y_col].max():.2f}")
    
    # 处理孕周数据（可能是字符串格式）
    try:
        ga_numeric = pd.to_numeric(df[ga_col], errors='coerce')
        print(f"  孕周: {ga_numeric.min():.2f} - {ga_numeric.max():.2f}")
    except:
        print(f"  孕周: {df[ga_col].min()} - {df[ga_col].max()} (字符串格式)")
    
    return True

def create_sample_data() -> pd.DataFrame:
    """
    创建示例数据用于测试
    """
    print("创建示例数据...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # 生成BMI数据
    bmi = np.random.normal(25, 5, n_samples)
    bmi = np.clip(bmi, 15, 45)
    
    # 生成孕周数据
    gestational_age = np.random.normal(18, 3, n_samples)
    gestational_age = np.clip(gestational_age, 10, 25)
    
    # 生成Y染色体浓度数据
    base_y = 8 + np.random.normal(0, 1, n_samples)
    bmi_effect = -0.1 * (bmi - 25)
    ga_effect = 0.2 * (gestational_age - 18)
    y_chromosome = base_y + bmi_effect + ga_effect + np.random.normal(0, 0.5, n_samples)
    y_chromosome = np.clip(y_chromosome, 2, 15)
    
    # 生成其他数据
    height = np.random.normal(165, 8, n_samples)
    weight = bmi * (height / 100) ** 2
    age = np.random.normal(30, 5, n_samples)
    age = np.clip(age, 20, 45)
    
    df = pd.DataFrame({
        '孕妇BMI': bmi,
        '检测孕周': gestational_age,
        'Y染色体浓度': y_chromosome,
        '身高': height,
        '体重': weight,
        '年龄': age,
        '原始读段数': np.random.randint(1000000, 5000000, n_samples),
        'GC含量': np.random.normal(0.45, 0.05, n_samples)
    })
    
    print(f"示例数据创建完成: {df.shape}")
    return df

def main():
    """
    主函数 - 测试数据加载
    """
    print("数据加载测试")
    print("="*50)
    
    # 尝试加载真实数据
    df = load_data_robust()
    
    if df is not None:
        # 识别列名
        column_mapping = identify_columns_robust(df)
        
        # 验证数据
        if validate_data(df, column_mapping):
            print("✓ 数据加载和验证成功")
            return df, column_mapping
        else:
            print("✗ 数据验证失败")
    else:
        print("✗ 无法加载真实数据，使用示例数据")
        df = create_sample_data()
        column_mapping = identify_columns_robust(df)
    
    return df, column_mapping

if __name__ == "__main__":
    df, column_mapping = main()
    if df is not None:
        print(f"\n最终数据形状: {df.shape}")
        print(f"列名映射: {column_mapping}")
