#!/bin/bash
# 激活虚拟环境的脚本
cd /home/kyunana/CUMCM2025-C
source venv/bin/activate
echo "✅ 虚拟环境已激活！"
echo "📦 可用的依赖包："
echo "   - pandas (数据处理)"
echo "   - numpy (数值计算)"
echo "   - matplotlib (绘图)"
echo "   - seaborn (统计绘图)"
echo "   - scipy (科学计算)"
echo "   - statsmodels (统计建模)"
echo "   - scikit-learn (机器学习)"
echo "   - pygam (广义加性模型)"
echo ""
echo "🚀 您现在可以运行 Python 脚本了！"
echo "💡 例如：python scripts/t1_analysis.py"
