#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMI分组可视化工具
交互式调整BMI分区，显示Y染色体浓度和孕周的散点图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, Button
import warnings
from typing import List, Tuple, Dict, Any
import json

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BMIGroupingVisualizer:
    """BMI分组可视化器"""
    
    def __init__(self, df: pd.DataFrame, bmi_col: str, y_col: str, ga_col: str):
        """
        初始化可视化器
        
        Args:
            df: 数据框
            bmi_col: BMI列名
            y_col: Y染色体浓度列名
            ga_col: 孕周列名
        """
        self.df = df.copy()
        self.bmi_col = bmi_col
        self.y_col = y_col
        self.ga_col = ga_col
        
        # 预处理数据
        self._preprocess_data()
        
        # 默认BMI分区阈值
        self.bmi_thresholds = [18.5, 25, 30, 35]
        self.bmi_labels = ['偏瘦', '正常', '超重', '肥胖I级', '肥胖II级']
        
        # 颜色方案
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # 创建分组
        self.update_groups()
    
    def _preprocess_data(self):
        """预处理数据"""
        # 转换孕周为数值格式
        if self.ga_col in self.df.columns:
            self.df[self.ga_col] = self._parse_gestational_age(self.df[self.ga_col])
        
        # 转换Y染色体浓度为百分比格式
        if self.y_col in self.df.columns:
            if self.df[self.y_col].max() <= 1:
                self.df[self.y_col] = self.df[self.y_col] * 100
    
    def _parse_gestational_age(self, ga_series):
        """解析孕周字符串"""
        def parse_ga(ga_str):
            try:
                if pd.isna(ga_str) or ga_str == '':
                    return np.nan
                ga_str = str(ga_str).strip()
                if 'w' in ga_str:
                    # 处理"11w+6"格式
                    parts = ga_str.split('w')
                    weeks = int(parts[0])
                    if '+' in parts[1]:
                        days = int(parts[1].split('+')[1])
                    else:
                        days = 0
                    return weeks + days / 7.0
                else:
                    # 直接转换为数值
                    return float(ga_str)
            except:
                return np.nan
        
        return ga_series.apply(parse_ga)
        
    def update_groups(self):
        """更新BMI分组"""
        # 创建BMI分组
        bins = [0] + self.bmi_thresholds + [100]
        self.df['bmi_group'] = pd.cut(
            self.df[self.bmi_col], 
            bins=bins, 
            labels=self.bmi_labels,
            include_lowest=True
        )
        
        # 计算各组的统计信息
        self.group_stats = self._calculate_group_stats()
    
    def _calculate_group_stats(self) -> Dict[str, Dict[str, Any]]:
        """计算各组的统计信息"""
        stats = {}
        
        for group in self.bmi_labels:
            group_data = self.df[self.df['bmi_group'] == group]
            
            if len(group_data) > 0:
                stats[group] = {
                    'count': len(group_data),
                    'bmi_mean': group_data[self.bmi_col].mean(),
                    'bmi_std': group_data[self.bmi_col].std(),
                    'y_mean': group_data[self.y_col].mean(),
                    'y_std': group_data[self.y_col].std(),
                    'ga_mean': group_data[self.ga_col].mean(),
                    'ga_std': group_data[self.ga_col].std(),
                    'y_ga_corr': group_data[self.y_col].corr(group_data[self.ga_col])
                }
            else:
                stats[group] = {
                    'count': 0,
                    'bmi_mean': np.nan,
                    'bmi_std': np.nan,
                    'y_mean': np.nan,
                    'y_std': np.nan,
                    'ga_mean': np.nan,
                    'ga_std': np.nan,
                    'y_ga_corr': np.nan
                }
        
        return stats
    
    def create_interactive_plot(self):
        """创建交互式图表"""
        # 创建图形和子图
        fig = plt.figure(figsize=(16, 12))
        
        # 主散点图
        self.ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        
        # 统计信息显示区域
        self.ax_stats = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        self.ax_stats.axis('off')
        
        # BMI分布直方图
        self.ax_hist = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        
        # 滑块区域
        self.ax_sliders = plt.subplot2grid((3, 3), (2, 2))
        self.ax_sliders.axis('off')
        
        # 绘制初始图表
        self.plot_scatter()
        self.plot_histogram()
        self.update_stats_display()
        
        # 创建滑块
        self.create_sliders()
        
        # 添加重置按钮
        self.add_reset_button()
        
        plt.tight_layout()
        plt.show()
    
    def plot_scatter(self):
        """绘制散点图"""
        self.ax_main.clear()
        
        # 为每个BMI分组绘制散点图
        for i, group in enumerate(self.bmi_labels):
            group_data = self.df[self.df['bmi_group'] == group]
            
            if len(group_data) > 0:
                # 绘制散点
                scatter = self.ax_main.scatter(
                    group_data[self.ga_col], 
                    group_data[self.y_col],
                    c=self.colors[i],
                    label=f'{group} (n={len(group_data)})',
                    alpha=0.7,
                    s=50,
                    edgecolors='white',
                    linewidth=0.5
                )
                
                # 添加趋势线
                if len(group_data) > 1:
                    try:
                        # 检查数据是否有效
                        ga_data = group_data[self.ga_col].dropna()
                        y_data = group_data[self.y_col].dropna()
                        
                        if len(ga_data) > 1 and len(y_data) > 1:
                            # 确保数据长度一致
                            min_len = min(len(ga_data), len(y_data))
                            ga_data = ga_data.iloc[:min_len]
                            y_data = y_data.iloc[:min_len]
                            
                            z = np.polyfit(ga_data, y_data, 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(ga_data.min(), ga_data.max(), 100)
                            self.ax_main.plot(x_trend, p(x_trend), color=self.colors[i], linestyle='--', alpha=0.8)
                    except:
                        # 如果趋势线计算失败，跳过
                        pass
        
        # 设置标签和标题
        self.ax_main.set_xlabel('孕周 (周)', fontsize=12)
        self.ax_main.set_ylabel('Y染色体浓度 (%)', fontsize=12)
        self.ax_main.set_title('BMI分组：Y染色体浓度 vs 孕周', fontsize=14, fontweight='bold')
        self.ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax_main.grid(True, alpha=0.3)
        
        # 设置坐标轴范围
        if len(self.df) > 0:
            self.ax_main.set_xlim(self.df[self.ga_col].min() - 0.5, self.df[self.ga_col].max() + 0.5)
            self.ax_main.set_ylim(self.df[self.y_col].min() - 0.5, self.df[self.y_col].max() + 0.5)
    
    def plot_histogram(self):
        """绘制BMI分布直方图"""
        self.ax_hist.clear()
        
        # 绘制BMI分布直方图
        self.ax_hist.hist(self.df[self.bmi_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 添加分区线
        for threshold in self.bmi_thresholds:
            self.ax_hist.axvline(threshold, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # 设置标签
        self.ax_hist.set_xlabel('BMI (kg/m²)', fontsize=12)
        self.ax_hist.set_ylabel('频数', fontsize=12)
        self.ax_hist.set_title('BMI分布及分区阈值', fontsize=12)
        self.ax_hist.grid(True, alpha=0.3)
    
    def update_stats_display(self):
        """更新统计信息显示"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # 创建统计信息文本
        stats_text = "BMI分组统计信息\n" + "="*30 + "\n\n"
        
        for i, group in enumerate(self.bmi_labels):
            stats = self.group_stats[group]
            
            if stats['count'] > 0:
                stats_text += f"{group}:\n"
                stats_text += f"  样本数: {stats['count']}\n"
                stats_text += f"  BMI: {stats['bmi_mean']:.1f}±{stats['bmi_std']:.1f}\n"
                stats_text += f"  Y染色体: {stats['y_mean']:.2f}±{stats['y_std']:.2f}%\n"
                stats_text += f"  孕周: {stats['ga_mean']:.1f}±{stats['ga_std']:.1f}周\n"
                stats_text += f"  相关性: {stats['y_ga_corr']:.3f}\n\n"
            else:
                stats_text += f"{group}: 无数据\n\n"
        
        # 添加总体统计
        stats_text += "总体统计:\n"
        stats_text += f"  总样本数: {len(self.df)}\n"
        stats_text += f"  BMI范围: {self.df[self.bmi_col].min():.1f}-{self.df[self.bmi_col].max():.1f}\n"
        stats_text += f"  Y染色体范围: {self.df[self.y_col].min():.2f}-{self.df[self.y_col].max():.2f}%\n"
        stats_text += f"  孕周范围: {self.df[self.ga_col].min():.1f}-{self.df[self.ga_col].max():.1f}周\n"
        
        # 显示文本
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes, 
                          fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    def create_sliders(self):
        """创建滑块"""
        # 计算BMI范围
        bmi_min = self.df[self.bmi_col].min()
        bmi_max = self.df[self.bmi_col].max()
        
        # 创建滑块位置
        slider_height = 0.8 / len(self.bmi_thresholds)
        
        self.sliders = []
        self.slider_axes = []
        
        for i, threshold in enumerate(self.bmi_thresholds):
            # 创建滑块轴
            ax_slider = plt.axes([0.7, 0.1 + i * slider_height, 0.25, slider_height * 0.8])
            self.slider_axes.append(ax_slider)
            
            # 创建滑块
            slider = Slider(
                ax_slider, 
                f'阈值{i+1}', 
                bmi_min, 
                bmi_max, 
                valinit=threshold,
                valfmt='%.1f'
            )
            
            # 设置滑块回调
            slider.on_changed(self.update_threshold)
            self.sliders.append(slider)
    
    def update_threshold(self, val):
        """更新阈值"""
        # 更新阈值
        for i, slider in enumerate(self.sliders):
            self.bmi_thresholds[i] = slider.val
        
        # 排序阈值
        self.bmi_thresholds.sort()
        
        # 更新分组和图表
        self.update_groups()
        self.plot_scatter()
        self.plot_histogram()
        self.update_stats_display()
        
        # 刷新图表
        plt.draw()
    
    def add_reset_button(self):
        """添加重置按钮"""
        ax_reset = plt.axes([0.7, 0.05, 0.1, 0.04])
        self.btn_reset = Button(ax_reset, '重置')
        self.btn_reset.on_clicked(self.reset_thresholds)
    
    def reset_thresholds(self, event):
        """重置阈值"""
        # 重置为默认值
        default_thresholds = [18.5, 25, 30, 35]
        
        for i, slider in enumerate(self.sliders):
            slider.reset()
        
        self.bmi_thresholds = default_thresholds.copy()
        self.update_groups()
        self.plot_scatter()
        self.plot_histogram()
        self.update_stats_display()
        plt.draw()
    
    def save_plot(self, filename: str = 'bmi_grouping_plot.png'):
        """保存图表"""
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {filename}")
    
    def export_group_data(self, filename: str = 'bmi_group_data.csv'):
        """导出分组数据"""
        # 添加分组信息到数据框
        export_df = self.df.copy()
        export_df['bmi_group'] = export_df['bmi_group'].astype(str)
        
        # 保存到CSV
        export_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"分组数据已保存到: {filename}")
    
    def generate_report(self, filename: str = 'bmi_grouping_report.txt'):
        """生成分析报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("BMI分组分析报告\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总样本数: {len(self.df)}\n")
            f.write(f"BMI分区阈值: {self.bmi_thresholds}\n\n")
            
            f.write("各分组详细统计:\n")
            f.write("-"*30 + "\n")
            
            for group in self.bmi_labels:
                stats = self.group_stats[group]
                f.write(f"\n{group}:\n")
                f.write(f"  样本数: {stats['count']}\n")
                f.write(f"  BMI均值±标准差: {stats['bmi_mean']:.2f}±{stats['bmi_std']:.2f}\n")
                f.write(f"  Y染色体浓度均值±标准差: {stats['y_mean']:.3f}±{stats['y_std']:.3f}%\n")
                f.write(f"  孕周均值±标准差: {stats['ga_mean']:.2f}±{stats['ga_std']:.2f}周\n")
                f.write(f"  Y染色体与孕周相关性: {stats['y_ga_corr']:.3f}\n")
            
            f.write(f"\n总体统计:\n")
            f.write(f"  BMI范围: {self.df[self.bmi_col].min():.2f}-{self.df[self.bmi_col].max():.2f}\n")
            f.write(f"  Y染色体浓度范围: {self.df[self.y_col].min():.3f}-{self.df[self.y_col].max():.3f}%\n")
            f.write(f"  孕周范围: {self.df[self.ga_col].min():.2f}-{self.df[self.ga_col].max():.2f}周\n")
        
        print(f"分析报告已保存到: {filename}")
    
    def create_separate_group_plots(self, save_plots: bool = True):
        """为每个BMI分组创建单独的图表"""
        print("正在创建各BMI分组的独立图表...")
        
        # 为每个有数据的BMI分组创建独立图表
        for i, group in enumerate(self.bmi_labels):
            group_data = self.df[self.df['bmi_group'] == group]
            
            if len(group_data) > 0:
                # 创建单个分组的图表
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'BMI分组: {group} (n={len(group_data)})', fontsize=16, fontweight='bold')
                
                # 1. 散点图
                ax1 = axes[0, 0]
                self._plot_single_group_scatter(ax1, group, group_data)
                
                # 2. 孕周分布直方图
                ax2 = axes[0, 1]
                self._plot_single_group_ga_histogram(ax2, group, group_data)
                
                # 3. Y染色体浓度分布直方图
                ax3 = axes[1, 0]
                self._plot_single_group_y_histogram(ax3, group, group_data)
                
                # 4. 统计信息
                ax4 = axes[1, 1]
                self._plot_single_group_stats(ax4, group, group_data)
                
                plt.tight_layout()
                
                if save_plots:
                    filename = f'bmi_group_{group}_analysis.png'
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"  {group} 分组图表已保存到: {filename}")
                
                plt.show()
    
    def _plot_single_group_scatter(self, ax, group, group_data):
        """绘制单个分组的散点图"""
        # 绘制散点
        ax.scatter(
            group_data[self.ga_col], 
            group_data[self.y_col],
            c=self.colors[self.bmi_labels.index(group)],
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )
        
        # 添加趋势线
        if len(group_data) > 1:
            try:
                ga_data = group_data[self.ga_col].dropna()
                y_data = group_data[self.y_col].dropna()
                
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
    
    def _plot_single_group_ga_histogram(self, ax, group, group_data):
        """绘制单个分组的孕周分布直方图"""
        ga_data = group_data[self.ga_col].dropna()
        
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
    
    def _plot_single_group_y_histogram(self, ax, group, group_data):
        """绘制单个分组的Y染色体浓度分布直方图"""
        y_data = group_data[self.y_col].dropna()
        
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
    
    def _plot_single_group_stats(self, ax, group, group_data):
        """绘制单个分组的统计信息"""
        ax.axis('off')
        
        # 计算统计信息
        stats = self.group_stats[group]
        
        # 创建统计信息文本
        stats_text = f"{group} 分组统计信息\n" + "="*30 + "\n\n"
        stats_text += f"样本数: {stats['count']}\n\n"
        stats_text += f"BMI统计:\n"
        stats_text += f"  均值: {stats['bmi_mean']:.2f}\n"
        stats_text += f"  标准差: {stats['bmi_std']:.2f}\n"
        stats_text += f"  范围: {group_data[self.bmi_col].min():.2f} - {group_data[self.bmi_col].max():.2f}\n\n"
        stats_text += f"Y染色体浓度统计:\n"
        stats_text += f"  均值: {stats['y_mean']:.2f}%\n"
        stats_text += f"  标准差: {stats['y_std']:.2f}%\n"
        stats_text += f"  范围: {group_data[self.y_col].min():.2f} - {group_data[self.y_col].max():.2f}%\n\n"
        stats_text += f"孕周统计:\n"
        stats_text += f"  均值: {stats['ga_mean']:.2f}周\n"
        stats_text += f"  标准差: {stats['ga_std']:.2f}周\n"
        stats_text += f"  范围: {group_data[self.ga_col].min():.2f} - {group_data[self.ga_col].max():.2f}周\n\n"
        stats_text += f"相关性:\n"
        stats_text += f"  Y染色体-孕周: {stats['y_ga_corr']:.3f}\n"
        
        # 显示文本
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')

def load_data_for_visualization():
    """加载数据用于可视化"""
    from data_loader import load_data_robust, identify_columns_robust, validate_data, create_sample_data
    
    # 使用健壮的数据加载函数
    df = load_data_robust()
    
    if df is not None:
        # 识别列名
        column_mapping = identify_columns_robust(df)
        
        # 验证数据
        if validate_data(df, column_mapping):
            print("✓ 数据加载和验证成功")
            return df, column_mapping
        else:
            print("✗ 数据验证失败，使用示例数据")
    else:
        print("✗ 无法加载真实数据，使用示例数据")
    
    # 创建示例数据
    df = create_sample_data()
    column_mapping = identify_columns_robust(df)
    return df, column_mapping

def main():
    """主函数"""
    print("BMI分组可视化工具")
    print("="*50)
    
    # 加载数据和列名映射
    result = load_data_for_visualization()
    if isinstance(result, tuple):
        df, column_mapping = result
    else:
        print("数据加载失败，程序退出")
        return
    
    # 显示数据基本信息
    print(f"\n数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 获取关键列
    bmi_col = column_mapping.get('bmi')
    y_col = column_mapping.get('y_chromosome')
    ga_col = column_mapping.get('gestational_age')
    
    if not all([bmi_col, y_col, ga_col]):
        print(f"未找到必要的列:")
        print(f"  BMI列: {bmi_col}")
        print(f"  Y染色体列: {y_col}")
        print(f"  孕周列: {ga_col}")
        return
    
    print(f"\n使用列:")
    print(f"  BMI列: {bmi_col}")
    print(f"  Y染色体列: {y_col}")
    print(f"  孕周列: {ga_col}")
    
    # 创建可视化器
    visualizer = BMIGroupingVisualizer(df, bmi_col, y_col, ga_col)
    
    # 创建交互式图表
    print("\n正在创建交互式图表...")
    visualizer.create_interactive_plot()
    
    # 创建各BMI分组的独立图表
    print("\n正在创建各BMI分组的独立图表...")
    visualizer.create_separate_group_plots()
    
    # 保存结果
    visualizer.save_plot('bmi_grouping_visualization.png')
    visualizer.export_group_data('bmi_group_data.csv')
    visualizer.generate_report('bmi_grouping_report.txt')
    
    print("\n可视化完成！")
    print("使用滑块调整BMI分区阈值，观察不同分组的分布特征")

if __name__ == "__main__":
    main()
