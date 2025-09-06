#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3路径修复脚本（高级版）
处理不同格式的路径设置，确保alpha和beta路线输出按前缀分类
"""

import os
import re
import glob

def fix_all_t3_scripts():
    """修复所有T3脚本的路径"""
    
    print("=== 开始修复T3路径（高级版）===")
    
    # 处理Alpha路线
    print("\n--- 修复Alpha路线 ---")
    alpha_scripts = glob.glob("scripts/T3/alpha/**/*.py", recursive=True)
    for script_path in alpha_scripts:
        fix_script_path_advanced(script_path, 'alpha')
    
    # 处理Beta路线
    print("\n--- 修复Beta路线 ---")
    beta_scripts = glob.glob("scripts/T3/beta/**/*.py", recursive=True)
    for script_path in beta_scripts:
        fix_script_path_advanced(script_path, 'beta')
    
    print("\n=== 路径修复完成 ===")

def fix_script_path_advanced(script_path, route_type):
    """高级修复单个脚本的路径"""
    print(f"修复: {script_path}")
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 备份原文件
        backup_path = script_path + '.backup2'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 提取版本号
        version_match = re.search(r'v(\d+\.\d+)', script_path)
        version = version_match.group(1) if version_match else '1.0'
        
        # 修改路径
        modified_content = modify_paths_advanced(content, route_type, version, script_path)
        
        # 写回修改后的内容
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"  ✅ 修复完成，备份文件: {backup_path}")
        
    except Exception as e:
        print(f"  ❌ 修复失败: {e}")

def modify_paths_advanced(content, route_type, version, script_path):
    """高级修改脚本中的路径"""
    
    # 1. 修改结果目录路径
    patterns_to_replace = [
        # 标准格式
        (r"results_dir = os\.path\.join\(project_root, 'results', 'T3', 'v\d+\.\d+'\)", 
         f"results_dir = os.path.join(project_root, 'results', 'T3', '{route_type}', 'v{version}')"),
        
        (r"results_dir = os\.path\.join\(project_root, 'results', 'T3', 'v\d+\.\d+_\w+'\)", 
         f"results_dir = os.path.join(project_root, 'results', 'T3', '{route_type}', 'v{version}')"),
        
        # 直接路径格式
        (r"os\.path\.join\(project_root, 'results', 'T3', 'v\d+\.\d+'\)", 
         f"os.path.join(project_root, 'results', 'T3', '{route_type}', 'v{version}')"),
        
        (r"os\.path\.join\(project_root, 'results', 'T3', 'v\d+\.\d+_\w+'\)", 
         f"os.path.join(project_root, 'results', 'T3', '{route_type}', 'v{version}')"),
        
        # 特殊格式（如Beta路线）
        (r'RESULTS_DIR = os\.path\.join\(BASE_DIR, "results_t3_v\d+\.\d+_\w+"\)', 
         f'RESULTS_DIR = os.path.join(BASE_DIR, "results", "T3", "{route_type}", "v{version}")'),
        
        (r'results_dir = os\.path\.join\(project_root, "results_t3_v\d+\.\d+_\w+"\)', 
         f'results_dir = os.path.join(project_root, "results", "T3", "{route_type}", "v{version}")'),
    ]
    
    for pattern, replacement in patterns_to_replace:
        content = re.sub(pattern, replacement, content)
    
    # 2. 修改输出文件名前缀
    prefix = f'T3_{route_type}_v{version}'
    
    # 替换各种文件名格式
    filename_patterns = [
        (r"'T3_v\d+", f"'{prefix}"),
        (r'"T3_v\d+', f'"{prefix}'),
        (r"'T3_\w+_v\d+", f"'{prefix}"),
        (r'"T3_\w+_v\d+', f'"{prefix}'),
        (r"'T3_\w+_\w+", f"'{prefix}"),
        (r'"T3_\w+_\w+', f'"{prefix}'),
    ]
    
    for pattern, replacement in filename_patterns:
        content = re.sub(pattern, replacement, content)
    
    # 3. 修改报告标题
    title_patterns = [
        (r'# T3 v\d+\.\d+', f'# {prefix}'),
        (r'T3 v\d+\.\d+', f'{prefix}'),
        (r'T3_\w+_v\d+', f'{prefix}'),
    ]
    
    for pattern, replacement in title_patterns:
        content = re.sub(pattern, replacement, content)
    
    # 4. 特殊处理：确保结果目录创建
    if 'os.makedirs(results_dir, exist_ok=True)' not in content:
        # 在结果目录使用前添加创建目录的代码
        content = re.sub(
            r'(results_dir = os\.path\.join\([^)]+\))',
            r'\1\n    os.makedirs(results_dir, exist_ok=True)',
            content
        )
    
    return content

def create_all_results_directories():
    """创建所有结果目录"""
    print("\n=== 创建结果目录结构 ===")
    
    # 获取所有版本
    alpha_versions = ['v1.0', 'v1.2', 'v1.3']
    beta_versions = ['v1.0', 'v1.2', 'v1.3', 'v1.4']
    
    for version in alpha_versions:
        dir_path = f'results/T3/alpha/{version}'
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    for version in beta_versions:
        dir_path = f'results/T3/beta/{version}'
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")

def test_script_paths():
    """测试脚本路径修改"""
    print("\n=== 测试脚本路径修改 ===")
    
    # 测试Alpha路线
    alpha_test = 'scripts/T3/alpha/v1.3/t3_simplified_integrated.py'
    if os.path.exists(alpha_test):
        with open(alpha_test, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'results/T3/alpha/v1.3' in content:
            print("✅ Alpha路线路径修改成功")
        else:
            print("❌ Alpha路线路径修改失败")
    
    # 测试Beta路线
    beta_test = 'scripts/T3/beta/t3_scripts_v1.4/t3_analysis_v1.4.py'
    if os.path.exists(beta_test):
        with open(beta_test, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'results/T3/beta/v1.4' in content:
            print("✅ Beta路线路径修改成功")
        else:
            print("❌ Beta路线路径修改失败")

def main():
    """主函数"""
    print("🚀 T3路径修复工具（高级版）")
    print("=" * 60)
    
    # 创建结果目录
    create_all_results_directories()
    
    # 修复脚本路径
    fix_all_t3_scripts()
    
    # 测试修改结果
    test_script_paths()
    
    print("\n✅ 所有路径修复完成！")
    print("\n📁 新的目录结构:")
    print("results/T3/")
    print("├── alpha/")
    print("│   ├── v1.0/")
    print("│   ├── v1.2/")
    print("│   └── v1.3/")
    print("└── beta/")
    print("    ├── v1.0/")
    print("    ├── v1.2/")
    print("    ├── v1.3/")
    print("    └── v1.4/")
    
    print("\n📝 输出文件前缀:")
    print("- Alpha路线: T3_alpha_v1.x")
    print("- Beta路线: T3_beta_v1.x")

if __name__ == "__main__":
    main()
