#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMI分组可视化工具测试脚本
生成模拟数据并测试可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bmi_grouping_simple import SimpleBMIGroupingVisualizer

def generate_mock_data(n_samples=1000):
    """生成模拟的男性胎儿数据"""
    np.random.seed(42)  # 设置随机种子确保结果可重复
    
    # 生成BMI数据 (正态分布，均值25，标准差5)
    bmi = np.random.normal(25, 5, n_samples)
    bmi = np.clip(bmi, 15, 45)  # 限制在合理范围内
    
    # 生成孕周数据 (正态分布，均值18，标准差3)
    gestational_age = np.random.normal(18, 3, n_samples)
    gestational_age = np.clip(gestational_age, 10, 25)  # 限制在合理范围内
    
    # 生成Y染色体浓度数据 (与BMI和孕周相关)
    # 基础浓度
    base_y = 8 + np.random.normal(0, 1, n_samples)
    
    # BMI效应 (BMI越高，Y染色体浓度略低)
    bmi_effect = -0.1 * (bmi - 25)
    
    # 孕周效应 (孕周越高，Y染色体浓度略高)
    ga_effect = 0.2 * (gestational_age - 18)
    
    # 组合效应
    y_chromosome = base_y + bmi_effect + ga_effect + np.random.normal(0, 0.5, n_samples)
    y_chromosome = np.clip(y_chromosome, 2, 15)  # 限制在合理范围内
    
    # 生成其他相关数据
    height = np.random.normal(165, 8, n_samples)  # 身高
    weight = bmi * (height / 100) ** 2  # 根据BMI和身高计算体重
    age = np.random.normal(30, 5, n_samples)  # 年龄
    age = np.clip(age, 20, 45)
    
    # 创建数据框
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
    
    return df

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试BMI分组可视化工具 ===")
    
    # 生成模拟数据
    print("1. 生成模拟数据...")
    df = generate_mock_data(1000)
    print(f"   生成了 {len(df)} 个样本")
    print(f"   数据列: {list(df.columns)}")
    
    # 显示数据基本信息
    print("\n2. 数据基本信息:")
    print(f"   BMI范围: {df['孕妇BMI'].min():.2f} - {df['孕妇BMI'].max():.2f}")
    print(f"   孕周范围: {df['检测孕周'].min():.2f} - {df['检测孕周'].max():.2f}")
    print(f"   Y染色体浓度范围: {df['Y染色体浓度'].min():.2f} - {df['Y染色体浓度'].max():.2f}")
    
    # 创建可视化器
    print("\n3. 创建可视化器...")
    visualizer = SimpleBMIGroupingVisualizer(
        df, 
        '孕妇BMI', 
        'Y染色体浓度', 
        '检测孕周'
    )
    
    # 打印统计信息
    print("\n4. 分组统计信息:")
    visualizer.print_group_statistics()
    
    # 创建图表
    print("\n5. 生成可视化图表...")
    visualizer.create_static_plot(save_plot=True)
    
    # 创建阈值对比图
    print("\n6. 生成阈值对比图...")
    threshold_sets = [
        [18.5, 25, 30, 35],  # 标准阈值
        [20, 25, 30, 35],    # 调整阈值1
        [18, 24, 28, 32],    # 调整阈值2
    ]
    visualizer.create_multiple_threshold_plots(threshold_sets)
    
    # 导出结果
    print("\n7. 导出结果...")
    visualizer.export_results('test_bmi_grouping_results.csv')
    
    print("\n=== 测试完成 ===")
    print("请查看生成的图表文件:")
    print("- bmi_grouping_analysis.png")
    print("- bmi_threshold_comparison.png")
    print("- test_bmi_grouping_results.csv")

def test_different_scenarios():
    """测试不同场景"""
    print("\n=== 测试不同数据场景 ===")
    
    scenarios = [
        ("正常分布", 1000),
        ("小样本", 100),
        ("大样本", 5000),
    ]
    
    for scenario_name, n_samples in scenarios:
        print(f"\n测试场景: {scenario_name} (n={n_samples})")
        
        # 生成数据
        df = generate_mock_data(n_samples)
        
        # 创建可视化器
        visualizer = SimpleBMIGroupingVisualizer(
            df, 
            '孕妇BMI', 
            'Y染色体浓度', 
            '检测孕周'
        )
        
        # 打印简要统计
        print(f"  总样本数: {len(df)}")
        for group in visualizer.bmi_labels:
            count = visualizer.group_stats[group]['count']
            print(f"  {group}: {count} 个样本")

def main():
    """主函数"""
    print("BMI分组可视化工具测试")
    print("="*50)
    
    try:
        # 测试基本功能
        test_basic_functionality()
        
        # 测试不同场景
        test_different_scenarios()
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
