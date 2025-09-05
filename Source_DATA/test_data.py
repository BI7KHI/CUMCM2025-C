#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据文件读取
"""

import pandas as pd
import os

def test_data_files():
    """测试数据文件"""
    print("当前目录:", os.getcwd())
    print("目录中的文件:")
    for file in os.listdir('.'):
        print(f"  {file}")
    
    # 测试读取Excel文件
    try:
        print("\n尝试读取data.xlsx...")
        df = pd.read_excel('data.xlsx')
        print(f"成功读取data.xlsx，形状: {df.shape}")
        print("列名:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        print("\n前5行:")
        print(df.head())
        return df
    except Exception as e:
        print(f"读取data.xlsx失败: {e}")
    
    # 测试读取CSV文件
    try:
        print("\n尝试读取dataA.csv...")
        df = pd.read_csv('dataA.csv')
        print(f"成功读取dataA.csv，形状: {df.shape}")
        print("列名:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        print("\n前5行:")
        print(df.head())
        return df
    except Exception as e:
        print(f"读取dataA.csv失败: {e}")
    
    try:
        print("\n尝试读取dataB.csv...")
        df = pd.read_csv('dataB.csv')
        print(f"成功读取dataB.csv，形状: {df.shape}")
        print("列名:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        print("\n前5行:")
        print(df.head())
        return df
    except Exception as e:
        print(f"读取dataB.csv失败: {e}")
    
    return None

if __name__ == "__main__":
    test_data_files()



