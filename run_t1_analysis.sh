#!/bin/bash

# T1分析脚本启动器
# 使用方法：./run_t1_analysis.sh

echo "正在启动T1分析脚本..."

# 切换到项目根目录
cd "$(dirname "$0")"

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 切换到T1分析脚本目录
cd scripts_v1.3_t1

# 运行T1分析脚本
echo "运行T1分析..."
python T1_analysis_v13.py

echo "T1分析完成！结果保存在 scripts_v1.3_t1/result_question1/ 目录下"
