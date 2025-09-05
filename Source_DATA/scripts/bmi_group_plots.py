#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMI分组独立图表生成器
为每个BMI分组创建单独的图表文件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from data_loader import load_data_robust, identify_columns_robust, validate_data

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_bmi_group_plots():
    """创建BMI分组独立图表"""
    print("BMI分组独立图表生成器")
    print("="*50)
    
    # 加载数据
    df = load_data_robust()
    if df is None:
        print("数据加载失败")
        return
    
    # 识别列名
    column_mapping = identify_columns_robust(df)
    
    # 验证数据
    if not validate_data(df, column_mapping):
        print("数据验证失败")
        return
    
    # 获取关键列
    bmi_col = column_mapping.get('bmi')
    y_col = column_mapping.get('y_chromosome')
    ga_col = column_mapping.get('gestational_age')
    
    if not all([bmi_col, y_col, ga_col]):
        print("未找到必要的列")
        return
    
    print(f"使用列: BMI={bmi_col}, Y染色体={y_col}, 孕周={ga_col}")
    
    # 预处理数据
    df = preprocess_data(df, bmi_col, y_col, ga_col)
    
    # 创建BMI分组
    bmi_thresholds = [18.5, 25, 30, 35]
    bmi_labels = ['偏瘦', '正常', '超重', '肥胖I级', '肥胖II级']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    bins = [0] + bmi_thresholds + [100]
    df['bmi_group'] = pd.cut(df[bmi_col], bins=bins, labels=bmi_labels, include_lowest=True)
    
    # 为每个分组创建图表
    for i, group in enumerate(bmi_labels):
        group_data = df[df['bmi_group'] == group]
        
        if len(group_data) > 0:
            print(f"\n正在创建 {group} 分组图表 (n={len(group_data)})...")
            create_single_group_plot(group, group_data, bmi_col, y_col, ga_col, colors[i])
    
    print("\n所有分组图表创建完成！")

def preprocess_data(df, bmi_col, y_col, ga_col):
    """预处理数据"""
    df = df.copy()
    
    # 转换孕周为数值格式
    if ga_col in df.columns:
        df[ga_col] = parse_gestational_age(df[ga_col])
    
    # 转换Y染色体浓度为百分比格式
    if y_col in df.columns:
        if df[y_col].max() <= 1:
            df[y_col] = df[y_col] * 100
    
    return df

def parse_gestational_age(ga_series):
    """解析孕周字符串"""
    def parse_ga(ga_str):
        try:
            if pd.isna(ga_str) or ga_str == '':
                return np.nan
            ga_str = str(ga_str).strip()
            if 'w' in ga_str:
                parts = ga_str.split('w')
                weeks = int(parts[0])
                if '+' in parts[1]:
                    days = int(parts[1].split('+')[1])
                else:
                    days = 0
                return weeks + days / 7.0
            else:
                return float(ga_str)
        except:
            return np.nan
    
    return ga_series.apply(parse_ga)

def create_single_group_plot(group, group_data, bmi_col, y_col, ga_col, color):
    """为单个分组创建图表"""
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'BMI分组: {group} (n={len(group_data)})', fontsize=16, fontweight='bold')
    
    # 1. 散点图
    ax1 = axes[0, 0]
    plot_scatter(ax1, group_data, ga_col, y_col, color, group)
    
    # 2. 孕周分布直方图
    ax2 = axes[0, 1]
    plot_ga_histogram(ax2, group_data, ga_col, group)
    
    # 3. Y染色体浓度分布直方图
    ax3 = axes[1, 0]
    plot_y_histogram(ax3, group_data, y_col, group)
    
    # 4. 统计信息
    ax4 = axes[1, 1]
    plot_statistics(ax4, group_data, bmi_col, y_col, ga_col, group)
    
    plt.tight_layout()
    
    # 保存图表
    filename = f'bmi_group_{group}_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  图表已保存到: {filename}")
    
    plt.close()  # 关闭图表以释放内存

def plot_scatter(ax, group_data, ga_col, y_col, color, group):
    """绘制散点图"""
    # 绘制散点
    ax.scatter(
        group_data[ga_col], 
        group_data[y_col],
        c=color,
        alpha=0.7,
        s=60,
        edgecolors='white',
        linewidth=0.5
    )
    
    # 添加趋势线
    if len(group_data) > 1:
        try:
            ga_data = group_data[ga_col].dropna()
            y_data = group_data[y_col].dropna()
            
            if len(ga_data) > 1 and len(y_data) > 1:
                min_len = min(len(ga_data), len(y_data))
                ga_data = ga_data.iloc[:min_len]
                y_data = y_data.iloc[:min_len]
                
                z = np.polyfit(ga_data, y_data, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(ga_data.min(), ga_data.max(), 100)
                ax.plot(x_trend, p(x_trend), color='red', linestyle='--', alpha=0.8, linewidth=2)
        except:
            pass
    
    # 设置标签
    ax.set_xlabel('孕周 (周)', fontsize=12)
    ax.set_ylabel('Y染色体浓度 (%)', fontsize=12)
    ax.set_title(f'{group} - Y染色体浓度 vs 孕周', fontsize=14)
    ax.grid(True, alpha=0.3)

def plot_ga_histogram(ax, group_data, ga_col, group):
    """绘制孕周分布直方图"""
    ga_data = group_data[ga_col].dropna()
    
    if len(ga_data) > 0:
        ax.hist(ga_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('孕周 (周)', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title(f'{group} - 孕周分布', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_ga = ga_data.mean()
        std_ga = ga_data.std()
        ax.axvline(mean_ga, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                  label=f'均值: {mean_ga:.1f}±{std_ga:.1f}')
        ax.legend()

def plot_y_histogram(ax, group_data, y_col, group):
    """绘制Y染色体浓度分布直方图"""
    y_data = group_data[y_col].dropna()
    
    if len(y_data) > 0:
        ax.hist(y_data, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.set_xlabel('Y染色体浓度 (%)', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title(f'{group} - Y染色体浓度分布', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_y = y_data.mean()
        std_y = y_data.std()
        ax.axvline(mean_y, color='red', linestyle='--', alpha=0.8, linewidth=2,
                  label=f'均值: {mean_y:.2f}±{std_y:.2f}')
        ax.legend()

def plot_statistics(ax, group_data, bmi_col, y_col, ga_col, group):
    """绘制统计信息"""
    ax.axis('off')
    
    # 计算统计信息
    stats_text = f"{group} 分组统计信息\n" + "="*30 + "\n\n"
    stats_text += f"样本数: {len(group_data)}\n\n"
    
    # BMI统计
    bmi_data = group_data[bmi_col].dropna()
    if len(bmi_data) > 0:
        stats_text += f"BMI统计:\n"
        stats_text += f"  均值: {bmi_data.mean():.2f}\n"
        stats_text += f"  标准差: {bmi_data.std():.2f}\n"
        stats_text += f"  范围: {bmi_data.min():.2f} - {bmi_data.max():.2f}\n\n"
    
    # Y染色体浓度统计
    y_data = group_data[y_col].dropna()
    if len(y_data) > 0:
        stats_text += f"Y染色体浓度统计:\n"
        stats_text += f"  均值: {y_data.mean():.2f}%\n"
        stats_text += f"  标准差: {y_data.std():.2f}%\n"
        stats_text += f"  范围: {y_data.min():.2f} - {y_data.max():.2f}%\n\n"
    
    # 孕周统计
    ga_data = group_data[ga_col].dropna()
    if len(ga_data) > 0:
        stats_text += f"孕周统计:\n"
        stats_text += f"  均值: {ga_data.mean():.2f}周\n"
        stats_text += f"  标准差: {ga_data.std():.2f}周\n"
        stats_text += f"  范围: {ga_data.min():.2f} - {ga_data.max():.2f}周\n\n"
    
    # 相关性
    if len(ga_data) > 1 and len(y_data) > 1:
        correlation = ga_data.corr(y_data)
        stats_text += f"相关性:\n"
        stats_text += f"  Y染色体-孕周: {correlation:.3f}\n"
    
    # 显示文本
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace')

if __name__ == "__main__":
    create_bmi_group_plots()
