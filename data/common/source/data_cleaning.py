#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
男性胎儿数据清洗脚本
基于T1建模意见进行数据清洗
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载数据文件"""
    print("正在加载数据文件...")
    
    # 尝试读取Excel文件
    try:
        df = pd.read_excel('data.xlsx')
        print(f"成功读取Excel文件，数据形状: {df.shape}")
        return df
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        # 尝试读取CSV文件
        try:
            df = pd.read_csv('dataA.csv')
            print(f"成功读取CSV文件A，数据形状: {df.shape}")
            return df
        except Exception as e:
            print(f"读取CSV文件A失败: {e}")
            try:
                df = pd.read_csv('dataB.csv')
                print(f"成功读取CSV文件B，数据形状: {df.shape}")
                return df
            except Exception as e:
                print(f"读取CSV文件B失败: {e}")
                return None

def explore_data(df):
    """探索数据结构"""
    print("\n=== 数据探索 ===")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n前5行数据:")
    print(df.head())
    print("\n数据类型:")
    print(df.dtypes)
    print("\n缺失值统计:")
    print(df.isnull().sum())
    print("\n基本统计信息:")
    print(df.describe())
    
    return df

def identify_columns(df):
    """识别关键列"""
    print("\n=== 列名识别 ===")
    
    # 根据实际数据结构直接映射
    column_mapping = {
        'fetal_sex': None,  # 通过Y染色体数据推断
        'y_chromosome': 'Y染色体浓度',
        'y_z_score': 'Y染色体的Z值', 
        'gestational_age': '检测孕周',
        'bmi': '孕妇BMI',
        'height': '身高',
        'weight': '体重',
        'age': '年龄',
        'fetal_health': '胎儿是否健康',
        'total_reads': '原始读段数',
        'gc_content': 'GC含量'
    }
    
    # 验证列是否存在
    for key, col_name in column_mapping.items():
        if col_name and col_name in df.columns:
            print(f"{key}: {col_name} ✓")
        else:
            print(f"{key}: {col_name} ✗ (未找到)")
            column_mapping[key] = None
    
    return column_mapping

def clean_male_fetal_data(df, column_mapping):
    """清洗男性胎儿数据"""
    print("\n=== 男性胎儿数据清洗 ===")
    
    # 1. 筛选男性胎儿样本
    print("1. 筛选男性胎儿样本...")
    
    # 根据Y染色体数据筛选男性胎儿
    y_col = column_mapping.get('y_chromosome')
    y_z_col = column_mapping.get('y_z_score')
    
    if y_col and y_col in df.columns:
        # 筛选Y染色体浓度非空的样本
        male_mask = df[y_col].notna()
        print(f"Y染色体浓度非空样本数: {male_mask.sum()}")
    elif y_z_col and y_z_col in df.columns:
        # 筛选Y染色体Z值非空的样本
        male_mask = df[y_z_col].notna()
        print(f"Y染色体Z值非空样本数: {male_mask.sum()}")
    else:
        print("未找到Y染色体相关列，无法筛选男性胎儿")
        return None
    
    df_male = df[male_mask].copy()
    print(f"筛选后男性胎儿样本数: {len(df_male)}")
    
    # 2. 处理核心变量缺失值
    print("\n2. 处理核心变量缺失值...")
    
    # 孕周缺失值处理
    ga_col = column_mapping.get('gestational_age')
    if ga_col and ga_col in df_male.columns:
        # 先转换孕周为数值类型
        print(f"孕周原始数据类型: {df_male[ga_col].dtype}")
        print(f"孕周样本值: {df_male[ga_col].head().tolist()}")
        
        # 尝试转换孕周为数值（处理"11w+6"格式）
        def parse_gestational_age(ga_str):
            """解析孕周字符串，如'11w+6' -> 11.86"""
            try:
                if pd.isna(ga_str) or ga_str == '':
                    return np.nan
                ga_str = str(ga_str).strip()
                if 'w' in ga_str:
                    # 处理"11w+6"格式
                    parts = ga_str.split('w')
                    weeks = int(parts[0])
                    if '+' in parts[1]:
                        days = int(parts[1].split('+')[1])
                    else:
                        days = 0
                    return weeks + days / 7.0
                else:
                    # 直接转换为数值
                    return float(ga_str)
            except:
                return np.nan
        
        df_male[ga_col] = df_male[ga_col].apply(parse_gestational_age)
        print("孕周已转换为数值类型")
        
        ga_missing = df_male[ga_col].isnull().sum()
        print(f"孕周缺失值数量: {ga_missing}")
        
        if ga_missing > 0:
            # 按BMI分组填充孕周中位数
            bmi_col = column_mapping.get('bmi')
            if bmi_col and bmi_col in df_male.columns:
                # 创建BMI分组
                df_male['bmi_group'] = pd.cut(df_male[bmi_col], 
                                            bins=[0, 20, 28, 35, 100], 
                                            labels=['<20', '20-28', '28-35', '≥35'])
                
                # 按BMI分组填充孕周中位数
                for group in df_male['bmi_group'].unique():
                    if pd.notna(group):
                        group_mask = df_male['bmi_group'] == group
                        ga_median = df_male.loc[group_mask & df_male[ga_col].notna(), ga_col].median()
                        if pd.notna(ga_median):
                            df_male.loc[group_mask & df_male[ga_col].isna(), ga_col] = ga_median
                            print(f"BMI分组 {group} 填充孕周中位数: {ga_median}")
            else:
                # 直接用总体中位数填充
                ga_median = df_male[ga_col].median()
                df_male[ga_col].fillna(ga_median, inplace=True)
                print(f"用总体中位数填充孕周: {ga_median}")
    
    # BMI缺失值处理
    bmi_col = column_mapping.get('bmi')
    if bmi_col and bmi_col in df_male.columns:
        bmi_missing = df_male[bmi_col].isnull().sum()
        print(f"BMI缺失值数量: {bmi_missing}")
        
        if bmi_missing > 0:
            # 尝试从身高体重计算BMI
            height_col = column_mapping.get('height')
            weight_col = column_mapping.get('weight')
            
            if height_col and weight_col and height_col in df_male.columns and weight_col in df_male.columns:
                # 计算BMI
                height_m = df_male[height_col] / 100  # 假设身高单位为cm
                df_male['bmi_calculated'] = df_male[weight_col] / (height_m ** 2)
                
                # 用计算值填充缺失的BMI
                df_male[bmi_col].fillna(df_male['bmi_calculated'], inplace=True)
                print(f"从身高体重计算BMI填充了 {bmi_missing} 个缺失值")
            else:
                # 用中位数填充
                bmi_median = df_male[bmi_col].median()
                df_male[bmi_col].fillna(bmi_median, inplace=True)
                print(f"用中位数填充BMI: {bmi_median}")
    
    # 3. 识别并剔除异常值
    print("\n3. 识别并剔除异常值...")
    
    original_count = len(df_male)
    
    # 孕周异常值
    if ga_col and ga_col in df_male.columns:
        # 剔除<10周或>25周的样本
        ga_outliers = (df_male[ga_col] < 10) | (df_male[ga_col] > 25)
        print(f"孕周异常值数量: {ga_outliers.sum()}")
        df_male = df_male[~ga_outliers]
    
    # BMI异常值
    if bmi_col and bmi_col in df_male.columns:
        # 剔除BMI<15或>60的样本
        bmi_outliers = (df_male[bmi_col] < 15) | (df_male[bmi_col] > 60)
        print(f"BMI异常值数量: {bmi_outliers.sum()}")
        df_male = df_male[~bmi_outliers]
    
    # Y染色体浓度异常值
    if y_col and y_col in df_male.columns:
        # 剔除<0%或>30%的异常值
        y_outliers = (df_male[y_col] < 0) | (df_male[y_col] > 30)
        print(f"Y染色体浓度异常值数量: {y_outliers.sum()}")
        df_male = df_male[~y_outliers]
    
    # 4. 数据质量控制
    print("\n4. 数据质量控制...")
    
    # 测序质量控制
    total_reads_col = column_mapping.get('total_reads')
    if total_reads_col and total_reads_col in df_male.columns:
        # 检查读段数分布
        reads_stats = df_male[total_reads_col].describe()
        print(f"读段数统计: {reads_stats}")
        
        # 使用更合理的阈值（1 million reads）
        low_reads = df_male[total_reads_col] < 1000000
        print(f"低测序深度样本数量: {low_reads.sum()}")
        df_male = df_male[~low_reads]
    
    # GC含量质量控制
    gc_col = column_mapping.get('gc_content')
    if gc_col and gc_col in df_male.columns:
        # 检查GC含量分布
        gc_stats = df_male[gc_col].describe()
        print(f"GC含量统计: {gc_stats}")
        
        # 使用更宽松的阈值（0.3-0.7，即30%-70%）
        gc_outliers = (df_male[gc_col] < 0.3) | (df_male[gc_col] > 0.7)
        print(f"GC含量异常样本数量: {gc_outliers.sum()}")
        df_male = df_male[~gc_outliers]
    
    # 5. 数据标准化
    print("\n5. 数据标准化...")
    
    # 孕周标准化（转换为decimal格式）
    if ga_col and ga_col in df_male.columns:
        # 假设孕周已经是decimal格式，如果不是需要转换
        print(f"孕周范围: {df_male[ga_col].min():.2f} - {df_male[ga_col].max():.2f} 周")
    
    # Y染色体浓度标准化（确保为百分比格式）
    if y_col and y_col in df_male.columns:
        # 如果Y染色体浓度是小数形式，转换为百分比
        if df_male[y_col].max() <= 1:
            df_male[y_col] = df_male[y_col] * 100
            print("Y染色体浓度已转换为百分比格式")
        print(f"Y染色体浓度范围: {df_male[y_col].min():.2f}% - {df_male[y_col].max():.2f}%")
    
    print(f"\n数据清洗完成！")
    print(f"原始样本数: {original_count}")
    print(f"清洗后样本数: {len(df_male)}")
    print(f"保留比例: {len(df_male)/original_count*100:.1f}%")
    
    return df_male

def generate_summary_report(df_original, df_cleaned, column_mapping):
    """生成数据清洗报告"""
    print("\n=== 数据清洗报告 ===")
    
    report = {
        '原始样本数': len(df_original),
        '清洗后样本数': len(df_cleaned),
        '保留比例': f"{len(df_cleaned)/len(df_original)*100:.1f}%"
    }
    
    # 核心变量统计
    ga_col = column_mapping.get('gestational_age')
    bmi_col = column_mapping.get('bmi')
    y_col = column_mapping.get('y_chromosome')
    
    if ga_col and ga_col in df_cleaned.columns:
        report['孕周范围'] = f"{df_cleaned[ga_col].min():.1f} - {df_cleaned[ga_col].max():.1f} 周"
        report['孕周中位数'] = f"{df_cleaned[ga_col].median():.1f} 周"
    
    if bmi_col and bmi_col in df_cleaned.columns:
        report['BMI范围'] = f"{df_cleaned[bmi_col].min():.1f} - {df_cleaned[bmi_col].max():.1f} kg/m²"
        report['BMI中位数'] = f"{df_cleaned[bmi_col].median():.1f} kg/m²"
        
        # BMI分组统计
        bmi_groups = pd.cut(df_cleaned[bmi_col], 
                           bins=[0, 18.5, 25, 30, 35, 100], 
                           labels=['偏瘦', '正常', '超重', '肥胖I级', '肥胖II级'])
        report['BMI分组分布'] = bmi_groups.value_counts().to_dict()
    
    if y_col and y_col in df_cleaned.columns:
        report['Y染色体浓度范围'] = f"{df_cleaned[y_col].min():.2f}% - {df_cleaned[y_col].max():.2f}%"
        report['Y染色体浓度中位数'] = f"{df_cleaned[y_col].median():.2f}%"
        
        # 检测失败率（<4%）
        low_fraction = (df_cleaned[y_col] < 4).sum()
        report['检测失败率(<4%)'] = f"{low_fraction/len(df_cleaned)*100:.1f}% ({low_fraction}例)"
    
    # 打印报告
    for key, value in report.items():
        print(f"{key}: {value}")
    
    return report

def save_cleaned_data(df_cleaned, column_mapping):
    """保存清洗后的数据"""
    print("\n=== 保存清洗后的数据 ===")
    
    # 保存为Excel文件
    output_file = 'male_fetal_data_cleaned.xlsx'
    df_cleaned.to_excel(output_file, index=False)
    print(f"已保存清洗后的数据到: {output_file}")
    
    # 保存为CSV文件
    output_csv = 'male_fetal_data_cleaned.csv'
    df_cleaned.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"已保存清洗后的数据到: {output_csv}")
    
    # 保存列名映射
    mapping_file = 'column_mapping.txt'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write("列名映射关系:\n")
        for key, value in column_mapping.items():
            f.write(f"{key}: {value}\n")
    print(f"已保存列名映射到: {mapping_file}")

def main():
    """主函数"""
    print("=== 男性胎儿数据清洗程序 ===")
    
    try:
        # 1. 加载数据
        df = load_data()
        if df is None:
            print("数据加载失败，程序退出")
            return
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 探索数据
    df = explore_data(df)
    
    # 3. 识别关键列
    column_mapping = identify_columns(df)
    
    # 4. 清洗男性胎儿数据
    df_cleaned = clean_male_fetal_data(df, column_mapping)
    if df_cleaned is None:
        print("数据清洗失败，程序退出")
        return
    
    # 5. 生成报告
    report = generate_summary_report(df, df_cleaned, column_mapping)
    
    # 6. 保存清洗后的数据
    save_cleaned_data(df_cleaned, column_mapping)
    
    print("\n=== 数据清洗完成 ===")

if __name__ == "__main__":
    main()
