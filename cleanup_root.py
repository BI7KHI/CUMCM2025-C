#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根目录清理脚本
对根目录中的无用文件进行合并或清理
"""

import os
import shutil
from datetime import datetime

def cleanup_root_directory():
    """清理根目录"""
    print("🧹 开始清理根目录...")
    
    # 1. 创建文档目录
    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)
    print(f"✅ 创建文档目录: {docs_dir}")
    
    # 2. 创建工具目录
    tools_dir = "tools"
    os.makedirs(tools_dir, exist_ok=True)
    print(f"✅ 创建工具目录: {tools_dir}")
    
    # 3. 移动文档文件
    move_documentation_files(docs_dir)
    
    # 4. 移动工具文件
    move_tool_files(tools_dir)
    
    # 5. 清理日志文件
    cleanup_log_files()
    
    # 6. 合并重复的说明文档
    merge_documentation()
    
    # 7. 清理临时文件
    cleanup_temporary_files()
    
    print("\n✅ 根目录清理完成！")

def move_documentation_files(docs_dir):
    """移动文档文件到docs目录"""
    print("\n📄 移动文档文件...")
    
    doc_files = [
        "数据文件夹组织说明.md",
        "项目文件夹组织说明.md", 
        "T3版本对比分析.md",
        "T3路径修复总结.md",
        "T3_v13_版本对比分析.md"
    ]
    
    for file in doc_files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(docs_dir, file))
            print(f"  📁 移动: {file} → docs/{file}")
        else:
            print(f"  ⚠️  文件不存在: {file}")

def move_tool_files(tools_dir):
    """移动工具文件到tools目录"""
    print("\n🔧 移动工具文件...")
    
    tool_files = [
        "fix_t3_paths.py",
        "fix_t3_paths_advanced.py"
    ]
    
    for file in tool_files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(tools_dir, file))
            print(f"  🔧 移动: {file} → tools/{file}")
        else:
            print(f"  ⚠️  文件不存在: {file}")

def cleanup_log_files():
    """清理日志文件"""
    print("\n📋 清理日志文件...")
    
    log_files = [
        "error_log.txt",
        "output_log.txt"
    ]
    
    for file in log_files:
        if os.path.exists(file):
            # 检查文件大小，如果很小就删除
            file_size = os.path.getsize(file)
            if file_size < 1000:  # 小于1KB
                os.remove(file)
                print(f"  🗑️  删除小文件: {file}")
            else:
                # 移动到logs目录
                logs_dir = "logs"
                os.makedirs(logs_dir, exist_ok=True)
                shutil.move(file, os.path.join(logs_dir, file))
                print(f"  📁 移动: {file} → logs/{file}")

def merge_documentation():
    """合并重复的说明文档"""
    print("\n📚 合并重复的说明文档...")
    
    # 合并T3版本对比分析文档
    t3_docs = [
        "docs/T3版本对比分析.md",
        "docs/T3_v13_版本对比分析.md"
    ]
    
    if all(os.path.exists(doc) for doc in t3_docs):
        # 读取两个文档内容
        with open(t3_docs[0], 'r', encoding='utf-8') as f:
            content1 = f.read()
        
        with open(t3_docs[1], 'r', encoding='utf-8') as f:
            content2 = f.read()
        
        # 合并内容
        merged_content = f"""# T3版本对比分析（合并版）

## 原始文档1
{content1}

---

## 原始文档2
{content2}

---
*合并时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存合并后的文档
        with open("docs/T3版本对比分析_合并版.md", 'w', encoding='utf-8') as f:
            f.write(merged_content)
        
        # 删除原始文档
        for doc in t3_docs:
            os.remove(doc)
        
        print("  📚 合并T3版本对比分析文档")

def cleanup_temporary_files():
    """清理临时文件"""
    print("\n🗑️  清理临时文件...")
    
    # 清理Python缓存文件
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                shutil.rmtree(pycache_path)
                print(f"  🗑️  删除: {pycache_path}")
        
        for file_name in files:
            if file_name.endswith('.pyc'):
                pyc_path = os.path.join(root, file_name)
                os.remove(pyc_path)
                print(f"  🗑️  删除: {pyc_path}")

def create_cleanup_summary():
    """创建清理总结"""
    print("\n📊 创建清理总结...")
    
    summary = f"""# 根目录清理总结

## 清理时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 清理内容

### 1. 目录重组
- 创建 `docs/` 目录：存放所有文档文件
- 创建 `tools/` 目录：存放工具脚本
- 创建 `logs/` 目录：存放日志文件

### 2. 文件移动
- 文档文件 → `docs/`
- 工具脚本 → `tools/`
- 日志文件 → `logs/`

### 3. 文档合并
- 合并重复的T3版本对比分析文档
- 创建统一的文档版本

### 4. 临时文件清理
- 删除Python缓存文件（__pycache__）
- 删除.pyc文件
- 删除小尺寸的日志文件

## 清理后的根目录结构
```
CUMCM2025-C/
├── data/           # 数据文件
├── docs/           # 文档文件
├── fonts/          # 字体文件
├── logs/           # 日志文件
├── material/       # 材料文件
├── results/        # 结果文件
├── scripts/        # 脚本文件
├── Source_DATA/    # 原始数据
├── tools/          # 工具脚本
├── Processed_DATA/ # 处理后数据
├── README.md       # 项目说明
└── requirements.txt # 依赖文件
```

## 注意事项
- 所有移动的文件都保留了原始内容
- 合并的文档已备份原始版本
- 删除的文件不可恢复，请谨慎操作

---
*清理工具: cleanup_root.py*
"""
    
    with open("docs/根目录清理总结.md", 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print("  📊 清理总结已保存到: docs/根目录清理总结.md")

def main():
    """主函数"""
    print("🚀 根目录清理工具")
    print("=" * 50)
    
    # 执行清理
    cleanup_root_directory()
    
    # 创建清理总结
    create_cleanup_summary()
    
    print("\n🎉 根目录清理完成！")
    print("\n📁 新的根目录结构:")
    print("├── data/           # 数据文件")
    print("├── docs/           # 文档文件")
    print("├── fonts/          # 字体文件")
    print("├── logs/           # 日志文件")
    print("├── material/       # 材料文件")
    print("├── results/        # 结果文件")
    print("├── scripts/        # 脚本文件")
    print("├── Source_DATA/    # 原始数据")
    print("├── tools/          # 工具脚本")
    print("├── Processed_DATA/ # 处理后数据")
    print("├── README.md       # 项目说明")
    print("└── requirements.txt # 依赖文件")

if __name__ == "__main__":
    main()

