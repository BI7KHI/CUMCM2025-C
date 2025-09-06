#!/usr/bin/env python
"""
运行脚本的包装器，确保使用正确的Python环境
"""
import sys
import subprocess
import os

# 确保使用虚拟环境的Python
venv_python = "/home/kyunana/CUMCM2025-C/venv/bin/python"

if sys.executable != venv_python:
    print(f"当前Python: {sys.executable}")
    print(f"切换到虚拟环境Python: {venv_python}")
    # 使用虚拟环境的Python重新运行脚本
    result = subprocess.run([venv_python, "scripts/t1_analysis.py"], 
                          cwd="/home/kyunana/CUMCM2025-C")
    sys.exit(result.returncode)
else:
    print(f"使用虚拟环境Python: {sys.executable}")
    # 直接导入并运行脚本
    exec(open("scripts/t1_analysis.py").read())
