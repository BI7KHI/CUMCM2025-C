#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径工具函数
用于统一管理项目中的文件路径，解决路径依赖问题
"""

import os
import sys
from pathlib import Path

def get_project_root():
    """获取项目根目录"""
    current_file = Path(__file__).resolve()
    
    # 从当前文件向上查找，直到找到包含Source_DATA目录的目录
    for parent in current_file.parents:
        if (parent / 'Source_DATA').exists():
            return parent
    
    # 如果没找到，返回当前文件的父目录
    return current_file.parent

def get_data_path(filename):
    """获取数据文件路径"""
    project_root = get_project_root()
    data_dir = project_root / 'Source_DATA'
    return data_dir / filename

def find_file(filename, search_dirs=None):
    """查找文件，尝试多个可能的路径"""
    if search_dirs is None:
        search_dirs = [
            '../Source_DATA',
            '../../Source_DATA',
            'Source_DATA',
            './Source_DATA'
        ]
    
    current_dir = Path(__file__).parent
    
    for search_dir in search_dirs:
        # 尝试相对路径
        path = current_dir / search_dir / filename
        if path.exists():
            return str(path.resolve())
        
        # 尝试绝对路径
        abs_path = Path(search_dir) / filename
        if abs_path.exists():
            return str(abs_path.resolve())
    
    # 如果都没找到，返回None
    return None

def load_data_file(filename, file_type='csv'):
    """加载数据文件"""
    import pandas as pd
    
    file_path = find_file(filename)
    
    if file_path is None:
        raise FileNotFoundError(f"无法找到文件: {filename}")
    
    print(f"正在从 {file_path} 加载数据...")
    
    try:
        if file_type.lower() == 'csv':
            return pd.read_csv(file_path)
        elif file_type.lower() == 'excel' or file_type.lower() == 'xlsx':
            return pd.read_excel(file_path)
        elif file_type.lower() == 'json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
    except Exception as e:
        raise Exception(f"加载文件失败: {e}")

def get_output_path(filename):
    """获取输出文件路径"""
    current_dir = Path(__file__).parent
    return current_dir / filename

def ensure_output_dir(filename):
    """确保输出目录存在"""
    output_path = get_output_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def test_paths():
    """测试路径函数"""
    print("=== 路径测试 ===")
    
    # 测试项目根目录
    project_root = get_project_root()
    print(f"项目根目录: {project_root}")
    
    # 测试数据路径
    data_path = get_data_path('male_fetal_data_cleaned.csv')
    print(f"数据文件路径: {data_path}")
    print(f"数据文件存在: {data_path.exists()}")
    
    # 测试文件查找
    found_file = find_file('male_fetal_data_cleaned.csv')
    print(f"找到的文件: {found_file}")
    
    # 测试输出路径
    output_path = get_output_path('test_output.png')
    print(f"输出文件路径: {output_path}")

if __name__ == "__main__":
    test_paths()
