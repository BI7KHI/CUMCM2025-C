#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3路径修复脚本
修改alpha和beta两条路线的路径，让输出按前缀分类
"""

import os
import re
import glob

def fix_script_paths():
    """修复脚本中的路径设置"""
    
    # 定义alpha和beta路线的脚本文件
    alpha_scripts = [
        'scripts/T3/alpha/v1.0/t3_analysis.py',
        'scripts/T3/alpha/v1.0/t3_enhanced_analysis.py',
        'scripts/T3/alpha/v1.2/t3_analysis_v12.py',
        'scripts/T3/alpha/v1.2/t3_final_analysis.py',
        'scripts/T3/alpha/v1.3/t3_integrated_analysis.py',
        'scripts/T3/alpha/v1.3/t3_simplified_integrated.py'
    ]
    
    beta_scripts = [
        'scripts/T3/beta/t3_scripts_v1.0/t3_analysis.py',
        'scripts/T3/beta/t3_scripts_v1.0/t3_enhanced_analysis.py',
        'scripts/T3/beta/t3_scripts_v1.2/t3_analysis_v1.2.py',
        'scripts/T3/beta/t3_scripts_v1.3/t3_analysis_v1.3.py',
        'scripts/T3/beta/t3_scripts_v1.4/t3_analysis_v1.4.py'
    ]
    
    print("=== 开始修复T3路径 ===")
    
    # 修复alpha路线
    print("\n--- 修复Alpha路线 ---")
    for script_path in alpha_scripts:
        if os.path.exists(script_path):
            fix_script_path(script_path, 'alpha')
        else:
            print(f"文件不存在: {script_path}")
    
    # 修复beta路线
    print("\n--- 修复Beta路线 ---")
    for script_path in beta_scripts:
        if os.path.exists(script_path):
            fix_script_path(script_path, 'beta')
        else:
            print(f"文件不存在: {script_path}")
    
    print("\n=== 路径修复完成 ===")

def fix_script_path(script_path, route_type):
    """修复单个脚本的路径"""
    print(f"修复: {script_path}")
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 备份原文件
        backup_path = script_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 修改路径
        modified_content = modify_paths(content, route_type, script_path)
        
        # 写回修改后的内容
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"  ✅ 修复完成，备份文件: {backup_path}")
        
    except Exception as e:
        print(f"  ❌ 修复失败: {e}")

def modify_paths(content, route_type, script_path):
    """修改脚本中的路径"""
    
    # 提取版本号
    version_match = re.search(r'v(\d+\.\d+)', script_path)
    version = version_match.group(1) if version_match else '1.0'
    
    # 修改结果路径
    old_patterns = [
        r"results_dir = os\.path\.join\(project_root, 'results', 'T3', 'v\d+\.\d+'\)",
        r"results_dir = os\.path\.join\(project_root, 'results', 'T3', 'v\d+\.\d+_\w+'\)",
        r"os\.path\.join\(project_root, 'results', 'T3', 'v\d+\.\d+'\)",
        r"os\.path\.join\(project_root, 'results', 'T3', 'v\d+\.\d+_\w+'\)"
    ]
    
    new_path = f"os.path.join(project_root, 'results', 'T3', '{route_type}', 'v{version}')"
    
    for pattern in old_patterns:
        content = re.sub(pattern, new_path, content)
    
    # 修改输出文件名前缀
    if route_type == 'alpha':
        prefix = 'T3_alpha'
    else:
        prefix = 'T3_beta'
    
    # 修改保存文件的代码
    save_patterns = [
        r"plt\.savefig\(os\.path\.join\(results_dir, 'T3_v\d+",
        r"with open\(os\.path\.join\(results_dir, 'T3_v\d+",
        r"json\.dump\(.*, f, indent=2, ensure_ascii=False, default=str\)"
    ]
    
    # 替换文件名前缀
    content = re.sub(r"'T3_v\d+", f"'{prefix}_v{version}", content)
    content = re.sub(r'"T3_v\d+', f'"{prefix}_v{version}', content)
    
    # 修改报告标题
    content = re.sub(r'# T3 v\d+\.\d+', f'# {prefix} v{version}', content)
    content = re.sub(r'T3 v\d+\.\d+', f'{prefix} v{version}', content)
    
    return content

def create_results_directories():
    """创建结果目录结构"""
    print("\n=== 创建结果目录结构 ===")
    
    # 创建alpha和beta的结果目录
    base_dirs = [
        'results/T3/alpha/v1.0',
        'results/T3/alpha/v1.2', 
        'results/T3/alpha/v1.3',
        'results/T3/beta/v1.0',
        'results/T3/beta/v1.2',
        'results/T3/beta/v1.3',
        'results/T3/beta/v1.4'
    ]
    
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")

def main():
    """主函数"""
    print("🚀 T3路径修复工具")
    print("=" * 50)
    
    # 创建结果目录
    create_results_directories()
    
    # 修复脚本路径
    fix_script_paths()
    
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

if __name__ == "__main__":
    main()
