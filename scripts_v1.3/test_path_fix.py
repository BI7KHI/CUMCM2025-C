#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试路径修复的简单脚本
"""

def test_path_utils():
    """测试路径工具函数"""
    print("=== 测试路径工具函数 ===")
    
    try:
        from path_utils import load_data_file, test_paths
        print("路径工具函数导入成功")
        
        # 运行路径测试
        test_paths()
        
        # 尝试加载数据
        print("\n=== 测试数据加载 ===")
        df = load_data_file('male_fetal_data_cleaned.csv', 'csv')
        print(f"数据加载成功，形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"路径工具函数测试失败: {e}")
        return False

def test_basic_imports():
    """测试基本库导入"""
    print("\n=== 测试基本库导入 ===")
    
    try:
        import pandas as pd
        print("✓ pandas 导入成功")
        
        import numpy as np
        print("✓ numpy 导入成功")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib 导入成功")
        
        from sklearn.ensemble import RandomForestRegressor
        print("✓ sklearn 导入成功")
        
        return True
        
    except Exception as e:
        print(f"基本库导入失败: {e}")
        return False

def test_enhanced_script():
    """测试增强版脚本的基本功能"""
    print("\n=== 测试增强版脚本 ===")
    
    try:
        # 导入增强版脚本的函数
        import sys
        sys.path.append('.')
        import t1_analysis_enhanced_v1_3 as enhanced
        load_cleaned_data = enhanced.load_cleaned_data
        
        # 测试数据加载
        df = load_cleaned_data()
        if df is not None:
            print(f"✓ 增强版脚本数据加载成功，形状: {df.shape}")
            return True
        else:
            print("✗ 增强版脚本数据加载失败")
            return False
            
    except Exception as e:
        print(f"增强版脚本测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试路径修复...")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        test_basic_imports,
        test_path_utils,
        test_enhanced_script
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试 {test.__name__} 出现异常: {e}")
            results.append(False)
    
    # 总结结果
    print("\n" + "=" * 50)
    print("测试结果总结:")
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✓ 所有测试通过！路径修复成功。")
    else:
        print("✗ 部分测试失败，需要进一步检查。")
    
    return passed == total

if __name__ == "__main__":
    main()
