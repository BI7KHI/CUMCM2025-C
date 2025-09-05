#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMI分组可视化工具 - 简化版
使用matplotlib基础功能，减少依赖
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import List, Tuple, Dict, Any

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimpleBMIGroupingVisualizer:
    """简化版BMI分组可视化器"""
    
    def __init__(self, df: pd.DataFrame, bmi_col: str, y_col: str, ga_col: str):
        """初始化可视化器"""
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
    
    def create_static_plot(self, save_plot: bool = True):
        """创建静态图表"""
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('BMI分组：Y染色体浓度与孕周关系分析', fontsize=16, fontweight='bold')
        
        # 1. 主散点图
        ax1 = axes[0, 0]
        self._plot_scatter(ax1)
        
        # 2. BMI分布直方图
        ax2 = axes[0, 1]
        self._plot_histogram(ax2)
        
        # 3. 分组统计柱状图
        ax3 = axes[1, 0]
        self._plot_group_stats(ax3)
        
        # 4. 相关性热图
        ax4 = axes[1, 1]
        self._plot_correlation(ax4)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('bmi_grouping_analysis.png', dpi=300, bbox_inches='tight')
            print("图表已保存到: bmi_grouping_analysis.png")
        
        plt.show()
    
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
    
    def _plot_scatter(self, ax):
        """绘制散点图"""
        # 为每个BMI分组绘制散点图
        for i, group in enumerate(self.bmi_labels):
            group_data = self.df[self.df['bmi_group'] == group]
            
            if len(group_data) > 0:
                # 绘制散点
                ax.scatter(
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
                            ax.plot(x_trend, p(x_trend), color=self.colors[i], linestyle='--', alpha=0.8)
                    except:
                        # 如果趋势线计算失败，跳过
                        pass
        
        # 设置标签和标题
        ax.set_xlabel('孕周 (周)', fontsize=12)
        ax.set_ylabel('Y染色体浓度 (%)', fontsize=12)
        ax.set_title('Y染色体浓度 vs 孕周 (按BMI分组)', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_histogram(self, ax):
        """绘制BMI分布直方图"""
        # 绘制BMI分布直方图
        ax.hist(self.df[self.bmi_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 添加分区线
        for i, threshold in enumerate(self.bmi_thresholds):
            ax.axvline(threshold, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(threshold, ax.get_ylim()[1]*0.8, f'{threshold:.1f}', 
                   rotation=90, verticalalignment='bottom', fontsize=10)
        
        # 设置标签
        ax.set_xlabel('BMI (kg/m²)', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title('BMI分布及分区阈值', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    def _plot_group_stats(self, ax):
        """绘制分组统计柱状图"""
        # 准备数据
        groups = []
        y_means = []
        ga_means = []
        
        for group in self.bmi_labels:
            stats = self.group_stats[group]
            if stats['count'] > 0:
                groups.append(group)
                y_means.append(stats['y_mean'])
                ga_means.append(stats['ga_mean'])
        
        if groups:
            x = np.arange(len(groups))
            width = 0.35
            
            # 绘制柱状图
            bars1 = ax.bar(x - width/2, y_means, width, label='Y染色体浓度 (%)', alpha=0.8)
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, ga_means, width, label='孕周 (周)', alpha=0.8, color='orange')
            
            # 设置标签
            ax.set_xlabel('BMI分组', fontsize=12)
            ax.set_ylabel('Y染色体浓度 (%)', fontsize=12)
            ax2.set_ylabel('孕周 (周)', fontsize=12)
            ax.set_title('各分组均值对比', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(groups, rotation=45)
            
            # 添加数值标签
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                height1 = bar1.get_height()
                height2 = bar2.get_height()
                ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.1,
                       f'{height1:.2f}', ha='center', va='bottom', fontsize=9)
                ax2.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.1,
                        f'{height2:.1f}', ha='center', va='bottom', fontsize=9)
            
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
    
    def _plot_correlation(self, ax):
        """绘制相关性热图"""
        # 计算各分组的相关性
        correlations = []
        group_names = []
        
        for group in self.bmi_labels:
            stats = self.group_stats[group]
            if stats['count'] > 0 and not np.isnan(stats['y_ga_corr']):
                correlations.append(stats['y_ga_corr'])
                group_names.append(group)
        
        if correlations:
            # 创建热图数据
            corr_matrix = np.array(correlations).reshape(-1, 1)
            
            # 绘制热图
            im = ax.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto')
            
            # 设置标签
            ax.set_xticks([0])
            ax.set_xticklabels(['Y染色体-孕周相关性'])
            ax.set_yticks(range(len(group_names)))
            ax.set_yticklabels(group_names)
            ax.set_title('各分组Y染色体与孕周相关性', fontsize=14)
            
            # 添加数值标签
            for i in range(len(group_names)):
                ax.text(0, i, f'{correlations[i]:.3f}', 
                       ha='center', va='center', fontweight='bold')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    def create_multiple_threshold_plots(self, threshold_sets: List[List[float]]):
        """创建多个阈值设置的对比图"""
        n_sets = len(threshold_sets)
        fig, axes = plt.subplots(2, n_sets, figsize=(5*n_sets, 10))
        if n_sets == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('不同BMI分区阈值对比', fontsize=16, fontweight='bold')
        
        for i, thresholds in enumerate(threshold_sets):
            # 临时更新阈值
            original_thresholds = self.bmi_thresholds.copy()
            self.bmi_thresholds = thresholds
            self.update_groups()
            
            # 绘制散点图
            ax1 = axes[0, i]
            self._plot_scatter(ax1)
            ax1.set_title(f'阈值: {thresholds}')
            
            # 绘制直方图
            ax2 = axes[1, i]
            self._plot_histogram(ax2)
            ax2.set_title(f'BMI分布 (阈值: {thresholds})')
            
            # 恢复原始阈值
            self.bmi_thresholds = original_thresholds
            self.update_groups()
        
        plt.tight_layout()
        plt.savefig('bmi_threshold_comparison.png', dpi=300, bbox_inches='tight')
        print("阈值对比图已保存到: bmi_threshold_comparison.png")
        plt.show()
    
    def print_group_statistics(self):
        """打印分组统计信息"""
        print("\nBMI分组统计信息")
        print("="*50)
        
        for group in self.bmi_labels:
            stats = self.group_stats[group]
            
            if stats['count'] > 0:
                print(f"\n{group}:")
                print(f"  样本数: {stats['count']}")
                print(f"  BMI均值±标准差: {stats['bmi_mean']:.2f}±{stats['bmi_std']:.2f}")
                print(f"  Y染色体浓度均值±标准差: {stats['y_mean']:.3f}±{stats['y_std']:.3f}%")
                print(f"  孕周均值±标准差: {stats['ga_mean']:.2f}±{stats['ga_std']:.2f}周")
                print(f"  Y染色体与孕周相关性: {stats['y_ga_corr']:.3f}")
            else:
                print(f"\n{group}: 无数据")
        
        print(f"\n总体统计:")
        print(f"  总样本数: {len(self.df)}")
        print(f"  BMI范围: {self.df[self.bmi_col].min():.2f}-{self.df[self.bmi_col].max():.2f}")
        print(f"  Y染色体浓度范围: {self.df[self.y_col].min():.3f}-{self.df[self.y_col].max():.3f}%")
        print(f"  孕周范围: {self.df[self.ga_col].min():.2f}-{self.df[self.ga_col].max():.2f}周")
    
    def export_results(self, filename: str = 'bmi_grouping_results.csv'):
        """导出结果"""
        # 添加分组信息
        export_df = self.df.copy()
        export_df['bmi_group'] = export_df['bmi_group'].astype(str)
        
        # 保存到CSV
        export_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"结果已保存到: {filename}")

def load_data():
    """加载数据"""
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
    print("BMI分组可视化工具 - 简化版")
    print("="*50)
    
    # 加载数据和列名映射
    result = load_data()
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
    visualizer = SimpleBMIGroupingVisualizer(df, bmi_col, y_col, ga_col)
    
    # 打印统计信息
    visualizer.print_group_statistics()
    
    # 创建静态图表
    print("\n正在创建可视化图表...")
    visualizer.create_static_plot()
    
    # 创建各BMI分组的独立图表
    print("\n正在创建各BMI分组的独立图表...")
    visualizer.create_separate_group_plots()
    
    # 创建不同阈值对比图
    print("\n正在创建阈值对比图...")
    threshold_sets = [
        [18.5, 25, 30, 35],  # 标准阈值
        [20, 25, 30, 35],    # 调整阈值1
        [18, 24, 28, 32],    # 调整阈值2
        [19, 26, 31, 36]     # 调整阈值3
    ]
    visualizer.create_multiple_threshold_plots(threshold_sets)
    
    # 导出结果
    visualizer.export_results()
    
    print("\n可视化完成！")
    print("请查看生成的图表文件了解不同BMI分组的特征")

if __name__ == "__main__":
    main()
