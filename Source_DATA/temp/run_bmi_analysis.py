#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMI分组分析运行脚本
一键运行BMI分组可视化分析
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """检查依赖包"""
    required_packages = ['pandas', 'numpy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def find_data_files():
    """查找数据文件"""
    data_files = []
    
    # 查找可能的数据文件
    possible_files = [
        'cleaned_data/male_fetal_data_cleaned.xlsx',
        'data.xlsx',
        'dataA.csv',
        'dataB.csv'
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            data_files.append(file_path)
    
    return data_files

def run_analysis():
    """运行分析"""
    print("BMI分组可视化分析")
    print("="*50)
    
    # 检查依赖
    print("1. 检查依赖包...")
    if not check_dependencies():
        return False
    print("   ✓ 依赖包检查通过")
    
    # 查找数据文件
    print("\n2. 查找数据文件...")
    data_files = find_data_files()
    if not data_files:
        print("   ✗ 未找到数据文件")
        print("   请确保以下文件之一存在:")
        print("   - cleaned_data/male_fetal_data_cleaned.xlsx")
        print("   - data.xlsx")
        print("   - dataA.csv")
        print("   - dataB.csv")
        return False
    
    print(f"   ✓ 找到数据文件: {data_files[0]}")
    
    # 运行简化版分析
    print("\n3. 运行BMI分组分析...")
    try:
        from bmi_grouping_simple import main as run_simple_analysis
        run_simple_analysis()
        print("   ✓ 分析完成")
        return True
    except Exception as e:
        print(f"   ✗ 分析失败: {e}")
        return False

def run_interactive_analysis():
    """运行交互式分析"""
    print("\n4. 运行交互式分析...")
    try:
        from bmi_grouping_visualization import main as run_interactive_analysis
        run_interactive_analysis()
        print("   ✓ 交互式分析完成")
        return True
    except Exception as e:
        print(f"   ✗ 交互式分析失败: {e}")
        print("   请尝试运行简化版分析")
        return False

def show_results():
    """显示结果文件"""
    print("\n5. 生成的结果文件:")
    
    result_files = [
        'bmi_grouping_analysis.png',
        'bmi_threshold_comparison.png',
        'bmi_grouping_results.csv',
        'bmi_grouping_visualization.png',
        'bmi_group_data.csv',
        'bmi_grouping_report.txt'
    ]
    
    for file_path in result_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"   ✓ {file_path} ({file_size} bytes)")
        else:
            print(f"   - {file_path} (未生成)")

def main():
    """主函数"""
    print("BMI分组可视化分析工具")
    print("="*50)
    
    # 运行分析
    success = run_analysis()
    
    if success:
        # 尝试运行交互式分析
        run_interactive_analysis()
        
        # 显示结果
        show_results()
        
        print("\n" + "="*50)
        print("分析完成！")
        print("请查看生成的图表文件了解BMI分组特征")
        print("\n主要输出文件:")
        print("- bmi_grouping_analysis.png: 综合分析图表")
        print("- bmi_threshold_comparison.png: 阈值对比图")
        print("- bmi_grouping_results.csv: 分组结果数据")
    else:
        print("\n分析失败，请检查错误信息并重试")

if __name__ == "__main__":
    main()
