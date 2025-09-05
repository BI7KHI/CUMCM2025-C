#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件整理和清理工具
"""

import os
import shutil
from pathlib import Path

def organize_files():
    """整理文件到不同目录"""
    print("开始整理文件...")
    
    # 创建目录结构
    directories = {
        'scripts': '核心脚本',
        'data': '数据文件', 
        'output': '输出结果',
        'docs': '文档说明',
        'temp': '临时文件',
        'archive': '归档文件'
    }
    
    for dir_name, description in directories.items():
        os.makedirs(dir_name, exist_ok=True)
        print(f"创建目录: {dir_name} ({description})")
    
    # 文件分类规则
    file_categories = {
        'scripts': [
            'bmi_group_plots.py',
            'bmi_grouping_simple.py', 
            'bmi_grouping_visualization.py',
            'data_loader.py',
            'data_cleaning.py',
            'data_cleaning_improved.py'
        ],
        'data': [
            'data.xlsx',
            'dataA.csv', 
            'dataB.csv',
            'male_fetal_data_cleaned.xlsx',
            'male_fetal_data_cleaned.csv'
        ],
        'output': [
            'bmi_group_*.png',
            'bmi_grouping_*.png',
            'bmi_group_data.csv',
            'bmi_grouping_report.txt',
            'column_mapping.txt'
        ],
        'docs': [
            '*.md',
            'config.json',
            'README_improved.md'
        ],
        'temp': [
            'test_*.py',
            'example_usage.py',
            'run_bmi_analysis.py',
            'simple_clean.py'
        ]
    }
    
    # 移动文件
    moved_files = 0
    for category, patterns in file_categories.items():
        for pattern in patterns:
            if '*' in pattern:
                # 处理通配符
                import glob
                files = glob.glob(pattern)
                for file in files:
                    if os.path.exists(file):
                        shutil.move(file, f"{category}/{file}")
                        print(f"移动: {file} -> {category}/")
                        moved_files += 1
            else:
                # 处理具体文件名
                if os.path.exists(pattern):
                    shutil.move(pattern, f"{category}/{pattern}")
                    print(f"移动: {pattern} -> {category}/")
                    moved_files += 1
    
    print(f"\n文件整理完成！共移动 {moved_files} 个文件")
    
    # 清理无用文件
    cleanup_unnecessary_files()

def cleanup_unnecessary_files():
    """清理无用文件"""
    print("\n开始清理无用文件...")
    
    # 要删除的文件模式
    patterns_to_delete = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.DS_Store',
        'Thumbs.db'
    ]
    
    deleted_count = 0
    for pattern in patterns_to_delete:
        if pattern == '__pycache__':
            if os.path.exists(pattern):
                shutil.rmtree(pattern)
                print(f"删除目录: {pattern}")
                deleted_count += 1
        else:
            import glob
            files = glob.glob(pattern)
            for file in files:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"删除文件: {file}")
                    deleted_count += 1
    
    print(f"清理完成！共删除 {deleted_count} 个文件/目录")

def create_clean_workspace():
    """创建干净的工作空间"""
    print("\n创建干净的工作空间...")
    
    # 创建新的工作目录
    workspace_dir = "BMI_Analysis_Workspace"
    os.makedirs(workspace_dir, exist_ok=True)
    
    # 复制必要文件到工作空间
    essential_files = [
        'data.xlsx',
        'data_loader.py'
    ]
    
    for file in essential_files:
        if os.path.exists(file):
            shutil.copy2(file, workspace_dir)
            print(f"复制到工作空间: {file}")
    
    print(f"工作空间创建完成: {workspace_dir}/")

if __name__ == "__main__":
    print("文件整理工具")
    print("="*50)
    
    organize_files()
    create_clean_workspace()
    
    print("\n文件整理完成！")
    print("建议使用 BMI_Analysis_Workspace 目录进行后续工作")
