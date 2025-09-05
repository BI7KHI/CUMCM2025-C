#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗脚本使用示例
演示如何使用改进版数据清洗脚本
"""

from data_cleaning_improved import DataCleaner, DataCleaningConfig
import json

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 使用默认配置
    cleaner = DataCleaner()
    
    # 加载和清洗数据
    df = cleaner.load_data()
    if df is not None:
        df_cleaned = cleaner.clean_male_fetal_data(df)
        if df_cleaned is not None:
            cleaner.save_cleaned_data(df_cleaned)
            print("数据清洗完成！")

def example_custom_config():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    # 创建自定义配置
    config = DataCleaningConfig(
        data_files=['data.xlsx'],  # 指定数据文件
        output_dir='custom_output',  # 自定义输出目录
        gestational_age_range=(12.0, 24.0),  # 自定义孕周范围
        bmi_range=(18.0, 35.0),  # 自定义BMI范围
        missing_value_strategy='knn',  # 使用KNN填充
        outlier_method='zscore',  # 使用Z-Score检测异常值
        zscore_threshold=2.5,  # 自定义Z-Score阈值
        save_plots=True,  # 保存图表
        log_level='DEBUG'  # 详细日志
    )
    
    # 使用自定义配置
    cleaner = DataCleaner(config)
    
    # 执行清洗
    df = cleaner.load_data()
    if df is not None:
        df_cleaned = cleaner.clean_male_fetal_data(df)
        if df_cleaned is not None:
            cleaner.save_cleaned_data(df_cleaned)
            print("自定义配置数据清洗完成！")

def example_config_file():
    """从配置文件加载示例"""
    print("\n=== 配置文件示例 ===")
    
    try:
        # 加载配置文件
        with open('config.json', 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 创建配置对象
        config = DataCleaningConfig(**config_dict['data_cleaning_config'])
        
        # 使用配置
        cleaner = DataCleaner(config)
        
        # 执行清洗
        df = cleaner.load_data()
        if df is not None:
            df_cleaned = cleaner.clean_male_fetal_data(df)
            if df_cleaned is not None:
                cleaner.save_cleaned_data(df_cleaned)
                print("配置文件数据清洗完成！")
                
    except FileNotFoundError:
        print("配置文件 config.json 不存在，请先创建配置文件")
    except Exception as e:
        print(f"加载配置文件失败: {e}")

def example_step_by_step():
    """分步执行示例"""
    print("\n=== 分步执行示例 ===")
    
    cleaner = DataCleaner()
    
    # 步骤1: 加载数据
    print("步骤1: 加载数据")
    df = cleaner.load_data()
    if df is None:
        print("数据加载失败")
        return
    
    # 步骤2: 探索数据
    print("步骤2: 探索数据")
    df = cleaner.explore_data(df)
    
    # 步骤3: 识别列名
    print("步骤3: 识别列名")
    column_mapping = cleaner.identify_columns(df)
    
    # 步骤4: 清洗数据
    print("步骤4: 清洗数据")
    df_cleaned = cleaner.clean_male_fetal_data(df)
    if df_cleaned is None:
        print("数据清洗失败")
        return
    
    # 步骤5: 生成可视化
    print("步骤5: 生成可视化")
    cleaner.generate_visualizations(df, df_cleaned)
    
    # 步骤6: 生成报告
    print("步骤6: 生成报告")
    report = cleaner.generate_summary_report(df, df_cleaned)
    
    # 步骤7: 保存数据
    print("步骤7: 保存数据")
    cleaner.save_cleaned_data(df_cleaned)
    
    print("分步执行完成！")

def main():
    """主函数"""
    print("数据清洗脚本使用示例")
    print("=" * 50)
    
    # 运行各种示例
    example_basic_usage()
    example_custom_config()
    example_config_file()
    example_step_by_step()
    
    print("\n所有示例执行完成！")

if __name__ == "__main__":
    main()
