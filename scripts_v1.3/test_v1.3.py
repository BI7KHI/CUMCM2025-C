#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1 分析代码 v1.3 - 测试版本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """主函数"""
    print("T1 分析代码 v1.3 - 测试版本")
    print("=" * 50)
    
    # 加载数据
    try:
        df = pd.read_csv('../Source_DATA/male_fetal_data_cleaned.csv')
        print(f"成功加载数据，形状: {df.shape}")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 简单分析
    print("\n=== 数据概况 ===")
    print(f"样本数: {len(df)}")
    print(f"Y染色体浓度范围: {df['Y染色体浓度'].min():.2f} - {df['Y染色体浓度'].max():.2f}")
    print(f"检测孕周范围: {df['检测孕周'].min():.2f} - {df['检测孕周'].max():.2f}")
    print(f"孕妇BMI范围: {df['孕妇BMI'].min():.2f} - {df['孕妇BMI'].max():.2f}")
    
    # 相关性分析
    print("\n=== 相关性分析 ===")
    corr_ga = df['检测孕周'].corr(df['Y染色体浓度'])
    corr_bmi = df['孕妇BMI'].corr(df['Y染色体浓度'])
    corr_age = df['年龄'].corr(df['Y染色体浓度'])
    
    print(f"孕周与Y染色体浓度相关性: {corr_ga:.4f}")
    print(f"BMI与Y染色体浓度相关性: {corr_bmi:.4f}")
    print(f"年龄与Y染色体浓度相关性: {corr_age:.4f}")
    
    # 创建简单可视化
    print("\n=== 创建可视化 ===")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 孕周散点图
    axes[0, 0].scatter(df['检测孕周'], df['Y染色体浓度'], alpha=0.6, s=20)
    axes[0, 0].set_xlabel('检测孕周 (周)')
    axes[0, 0].set_ylabel('Y染色体浓度 (%)')
    axes[0, 0].set_title('Y染色体浓度与孕周关系')
    axes[0, 0].grid(True, alpha=0.3)
    
    # BMI散点图
    axes[0, 1].scatter(df['孕妇BMI'], df['Y染色体浓度'], alpha=0.6, s=20, color='orange')
    axes[0, 1].set_xlabel('孕妇BMI (kg/m²)')
    axes[0, 1].set_ylabel('Y染色体浓度 (%)')
    axes[0, 1].set_title('Y染色体浓度与BMI关系')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 年龄散点图
    axes[1, 0].scatter(df['年龄'], df['Y染色体浓度'], alpha=0.6, s=20, color='green')
    axes[1, 0].set_xlabel('年龄 (岁)')
    axes[1, 0].set_ylabel('Y染色体浓度 (%)')
    axes[1, 0].set_title('Y染色体浓度与年龄关系')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Y染色体浓度分布
    axes[1, 1].hist(df['Y染色体浓度'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(x=4, color='red', linestyle='--', label='NIPT失败阈值 (4%)')
    axes[1, 1].set_xlabel('Y染色体浓度 (%)')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('Y染色体浓度分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_visualization_v1.3.png', dpi=300, bbox_inches='tight')
    print("测试可视化图表已保存为: test_visualization_v1.3.png")
    
    # 生成简单报告
    report = f"""# T1分析测试报告 v1.3

## 数据概况
- 样本数: {len(df)}
- Y染色体浓度范围: {df['Y染色体浓度'].min():.2f} - {df['Y染色体浓度'].max():.2f}
- 检测孕周范围: {df['检测孕周'].min():.2f} - {df['检测孕周'].max():.2f}
- 孕妇BMI范围: {df['孕妇BMI'].min():.2f} - {df['孕妇BMI'].max():.2f}

## 相关性分析
- 孕周与Y染色体浓度相关性: {corr_ga:.4f}
- BMI与Y染色体浓度相关性: {corr_bmi:.4f}
- 年龄与Y染色体浓度相关性: {corr_age:.4f}

## 结论
测试版本运行成功，数据加载和基本分析正常。
"""
    
    with open('test_report_v1.3.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("测试报告已保存为: test_report_v1.3.md")
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()
