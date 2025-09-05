#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试中文字体显示
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def test_chinese_font():
    """测试中文字体显示"""
    print("=== 测试中文字体显示 ===")
    
    # 查找系统中可用的中文字体
    chinese_fonts = []
    for font in fm.fontManager.ttflist:
        font_name = font.name.lower()
        if any(name in font_name for name in ['microsoft', 'simhei', 'simsun', 'kaiti', 'fangsong', 'wenquanyi', 'noto', 'source han']):
            chinese_fonts.append(font.name)
    
    print(f"找到的中文字体: {chinese_fonts[:5]}")
    
    # 设置字体优先级
    if chinese_fonts:
        font_priority = chinese_fonts[:3] + ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        print(f"使用字体: {font_priority[:3]}")
    else:
        font_priority = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        print("未找到中文字体，使用默认字体")
    
    # 设置matplotlib字体
    plt.rcParams['font.sans-serif'] = font_priority
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 生成测试数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # 绘制图表
    ax.plot(x, y, 'b-', linewidth=2, label='正弦波')
    ax.set_xlabel('检测孕周 (周)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y染色体浓度 (%)', fontsize=14, fontweight='bold')
    ax.set_title('中文字体测试 - 统一算法评估系统', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加中文文本
    ax.text(5, 0.5, '这是中文测试文本', fontsize=14, ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 保存图表
    try:
        plt.savefig('chinese_font_test.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print("中文字体测试图表已保存为: chinese_font_test.png")
    except Exception as e:
        print(f"保存图表时出错: {e}")
    
    # 显示图表
    try:
        plt.show()
    except Exception as e:
        print(f"显示图表时出错: {e}")
    
    plt.close()

if __name__ == "__main__":
    test_chinese_font()
