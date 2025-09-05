#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的路径修复测试
"""

print("开始测试路径修复...")

# 测试基本库导入
try:
    import pandas as pd
    print("✓ pandas 导入成功")
except Exception as e:
    print(f"✗ pandas 导入失败: {e}")

try:
    import numpy as np
    print("✓ numpy 导入成功")
except Exception as e:
    print(f"✗ numpy 导入失败: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib 导入成功")
except Exception as e:
    print(f"✗ matplotlib 导入失败: {e}")

# 测试路径工具函数
try:
    from path_utils import load_data_file
    print("✓ path_utils 导入成功")
    
    # 测试数据加载
    df = load_data_file('male_fetal_data_cleaned.csv', 'csv')
    print(f"✓ 数据加载成功，形状: {df.shape}")
    print(f"✓ 列名: {list(df.columns)[:5]}...")  # 只显示前5列
    
except Exception as e:
    print(f"✗ path_utils 测试失败: {e}")

# 测试增强版脚本的数据加载函数
try:
    # 直接执行增强版脚本的数据加载部分
    exec(open('t1_analysis_enhanced_v1.3.py').read().split('def load_cleaned_data():')[1].split('def explore_cleaned_data')[0])
    print("✓ 增强版脚本数据加载函数测试成功")
except Exception as e:
    print(f"✗ 增强版脚本测试失败: {e}")

print("测试完成！")
