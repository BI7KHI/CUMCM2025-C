#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMI分组自动优化工具
通过迭代优化BMI分组阈值，使各组的相关性达到最高
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from itertools import combinations
import warnings
from data_loader import load_data_robust, identify_columns_robust, validate_data

warnings.filterwarnings('ignore')

# 修复中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class BMIOptimizer:
    """BMI分组优化器"""
    
    def __init__(self, df, bmi_col, y_col, ga_col):
        """初始化优化器"""
        self.df = df.copy()
        self.bmi_col = bmi_col
        self.y_col = y_col
        self.ga_col = ga_col
        
        # 预处理数据
        self._preprocess_data()
        
        # 优化结果存储
        self.optimization_history = []
        self.best_thresholds = None
        self.best_score = -np.inf
        
    def _preprocess_data(self):
        """预处理数据"""
        # 转换孕周为数值格式
        if self.ga_col in self.df.columns:
            self.df[self.ga_col] = self._parse_gestational_age(self.df[self.ga_col])
        
        # 转换Y染色体浓度为百分比格式
        if self.y_col in self.df.columns:
            if self.df[self.y_col].max() <= 1:
                self.df[self.y_col] = self.df[self.y_col] * 100
        
        # 移除缺失值
        self.df = self.df.dropna(subset=[self.bmi_col, self.y_col, self.ga_col])
        
        print(f"预处理后数据形状: {self.df.shape}")
    
    def _parse_gestational_age(self, ga_series):
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
    
    def calculate_group_correlations(self, thresholds):
        """计算各分组的平均相关性"""
        try:
            # 创建分组
            bins = [0] + sorted(thresholds) + [100]
            labels = [f'Group_{i+1}' for i in range(len(thresholds) + 1)]
            
            self.df['temp_group'] = pd.cut(
                self.df[self.bmi_col], 
                bins=bins, 
                labels=labels,
                include_lowest=True
            )
            
            correlations = []
            group_sizes = []
            
            for group in labels:
                group_data = self.df[self.df['temp_group'] == group]
                
                if len(group_data) >= 3:  # 至少需要3个样本计算相关性
                    corr = group_data[self.ga_col].corr(group_data[self.y_col])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))  # 使用绝对相关性
                        group_sizes.append(len(group_data))
            
            if len(correlations) == 0:
                return 0
            
            # 加权平均相关性（按样本数加权）
            if group_sizes:
                weights = np.array(group_sizes) / sum(group_sizes)
                weighted_corr = np.average(correlations, weights=weights)
                return weighted_corr
            else:
                return 0
                
        except Exception as e:
            return 0
    
    def objective_function(self, thresholds):
        """目标函数：最大化平均相关性"""
        # 确保阈值在合理范围内
        if any(t < 15 or t > 40 for t in thresholds):
            return -1000
        
        # 确保阈值递增
        if not all(thresholds[i] < thresholds[i+1] for i in range(len(thresholds)-1)):
            return -1000
        
        score = self.calculate_group_correlations(thresholds)
        
        # 记录优化历史
        self.optimization_history.append({
            'thresholds': thresholds.copy(),
            'score': score
        })
        
        return -score  # 最小化负相关性
    
    def optimize_thresholds(self, initial_thresholds=None, max_iterations=100):
        """优化BMI分组阈值"""
        print("开始优化BMI分组阈值...")
        
        if initial_thresholds is None:
            initial_thresholds = [18.5, 25, 30, 35]
        
        # 设置约束条件
        bmi_min = self.df[self.bmi_col].min()
        bmi_max = self.df[self.bmi_col].max()
        
        bounds = [(bmi_min + 1, bmi_max - 1) for _ in range(len(initial_thresholds))]
        
        # 使用多种优化方法
        methods = ['L-BFGS-B', 'TNC', 'SLSQP']
        best_result = None
        best_score = -np.inf
        
        for method in methods:
            try:
                print(f"尝试优化方法: {method}")
                result = minimize(
                    self.objective_function,
                    initial_thresholds,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': max_iterations}
                )
                
                if result.success and -result.fun > best_score:
                    best_result = result
                    best_score = -result.fun
                    print(f"  方法 {method} 成功，得分: {best_score:.4f}")
                else:
                    print(f"  方法 {method} 失败或得分较低")
                    
            except Exception as e:
                print(f"  方法 {method} 出错: {e}")
        
        if best_result is not None:
            self.best_thresholds = best_result.x
            self.best_score = best_score
            print(f"\n优化完成！")
            print(f"最佳阈值: {self.best_thresholds}")
            print(f"最佳得分: {self.best_score:.4f}")
        else:
            print("优化失败，使用初始阈值")
            self.best_thresholds = initial_thresholds
            self.best_score = self.calculate_group_correlations(initial_thresholds)
        
        return self.best_thresholds, self.best_score
    
    def grid_search_optimization(self, threshold_ranges=None):
        """网格搜索优化"""
        print("开始网格搜索优化...")
        
        if threshold_ranges is None:
            bmi_min = self.df[self.bmi_col].min()
            bmi_max = self.df[self.bmi_col].max()
            threshold_ranges = [
                np.linspace(bmi_min + 1, 22, 5),
                np.linspace(22, 28, 5), 
                np.linspace(28, 35, 5),
                np.linspace(35, bmi_max - 1, 5)
            ]
        
        best_score = -np.inf
        best_thresholds = None
        
        # 生成所有可能的阈值组合
        for t1 in threshold_ranges[0]:
            for t2 in threshold_ranges[1]:
                for t3 in threshold_ranges[2]:
                    for t4 in threshold_ranges[3]:
                        if t1 < t2 < t3 < t4:
                            thresholds = [t1, t2, t3, t4]
                            score = self.calculate_group_correlations(thresholds)
                            
                            if score > best_score:
                                best_score = score
                                best_thresholds = thresholds
                                print(f"发现更好的阈值: {thresholds}, 得分: {score:.4f}")
        
        self.best_thresholds = best_thresholds
        self.best_score = best_score
        
        print(f"\n网格搜索完成！")
        print(f"最佳阈值: {self.best_thresholds}")
        print(f"最佳得分: {self.best_score:.4f}")
        
        return self.best_thresholds, self.best_score
    
    def create_optimized_plots(self, save_plots=True):
        """创建优化后的图表"""
        if self.best_thresholds is None:
            print("请先运行优化")
            return
        
        print("创建优化后的可视化图表...")
        
        # 创建分组
        bins = [0] + sorted(self.best_thresholds) + [100]
        labels = [f'Group_{i+1}' for i in range(len(self.best_thresholds) + 1)]
        
        self.df['optimized_group'] = pd.cut(
            self.df[self.bmi_col], 
            bins=bins, 
            labels=labels,
            include_lowest=True
        )
        
        # 创建主图表
        self._create_main_optimization_plot(save_plots)
        
        # 创建各分组独立图表
        self._create_group_plots(save_plots)
        
        # 创建优化历史图表
        self._create_optimization_history_plot(save_plots)
    
    def _create_main_optimization_plot(self, save_plots=True):
        """创建主优化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('BMI分组优化结果', fontsize=16, fontweight='bold')
        
        # 1. 散点图
        ax1 = axes[0, 0]
        self._plot_optimized_scatter(ax1)
        
        # 2. BMI分布
        ax2 = axes[0, 1]
        self._plot_bmi_distribution(ax2)
        
        # 3. 分组统计
        ax3 = axes[1, 0]
        self._plot_group_statistics(ax3)
        
        # 4. 相关性对比
        ax4 = axes[1, 1]
        self._plot_correlation_comparison(ax4)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('bmi_optimization_results.png', dpi=300, bbox_inches='tight')
            print("主优化图表已保存: bmi_optimization_results.png")
        
        plt.show()
    
    def _plot_optimized_scatter(self, ax):
        """绘制优化后的散点图"""
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, group in enumerate(self.df['optimized_group'].cat.categories):
            group_data = self.df[self.df['optimized_group'] == group]
            
            if len(group_data) > 0:
                ax.scatter(
                    group_data[self.ga_col], 
                    group_data[self.y_col],
                    c=colors[i % len(colors)],
                    label=f'{group} (n={len(group_data)})',
                    alpha=0.7,
                    s=50
                )
                
                # 添加趋势线
                if len(group_data) > 1:
                    try:
                        z = np.polyfit(group_data[self.ga_col], group_data[self.y_col], 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(group_data[self.ga_col].min(), group_data[self.ga_col].max(), 100)
                        ax.plot(x_trend, p(x_trend), color=colors[i % len(colors)], 
                               linestyle='--', alpha=0.8)
                    except:
                        pass
        
        ax.set_xlabel('孕周 (周)', fontsize=12)
        ax.set_ylabel('Y染色体浓度 (%)', fontsize=12)
        ax.set_title('优化后分组散点图', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_bmi_distribution(self, ax):
        """绘制BMI分布"""
        ax.hist(self.df[self.bmi_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 添加优化后的阈值线
        for threshold in self.best_thresholds:
            ax.axvline(threshold, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('BMI (kg/m²)', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title('BMI分布及优化阈值', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    def _plot_group_statistics(self, ax):
        """绘制分组统计"""
        group_stats = []
        group_names = []
        
        for group in self.df['optimized_group'].cat.categories:
            group_data = self.df[self.df['optimized_group'] == group]
            
            if len(group_data) > 0:
                corr = group_data[self.ga_col].corr(group_data[self.y_col])
                group_stats.append(abs(corr) if not np.isnan(corr) else 0)
                group_names.append(f'{group}\n(n={len(group_data)})')
        
        bars = ax.bar(group_names, group_stats, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(group_stats)])
        ax.set_ylabel('|相关性|', fontsize=12)
        ax.set_title('各分组相关性', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, stat in zip(bars, group_stats):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{stat:.3f}', ha='center', va='bottom')
    
    def _plot_correlation_comparison(self, ax):
        """绘制相关性对比"""
        # 原始分组 vs 优化分组
        original_thresholds = [18.5, 25, 30, 35]
        
        # 计算原始分组相关性
        bins_orig = [0] + original_thresholds + [100]
        labels_orig = [f'Orig_{i+1}' for i in range(len(original_thresholds) + 1)]
        self.df['original_group'] = pd.cut(
            self.df[self.bmi_col], 
            bins=bins_orig, 
            labels=labels_orig,
            include_lowest=True
        )
        
        orig_corrs = []
        opt_corrs = []
        
        for i, (orig_group, opt_group) in enumerate(zip(labels_orig, self.df['optimized_group'].cat.categories)):
            orig_data = self.df[self.df['original_group'] == orig_group]
            opt_data = self.df[self.df['optimized_group'] == opt_group]
            
            if len(orig_data) > 1:
                orig_corr = orig_data[self.ga_col].corr(orig_data[self.y_col])
                orig_corrs.append(abs(orig_corr) if not np.isnan(orig_corr) else 0)
            else:
                orig_corrs.append(0)
            
            if len(opt_data) > 1:
                opt_corr = opt_data[self.ga_col].corr(opt_data[self.y_col])
                opt_corrs.append(abs(opt_corr) if not np.isnan(opt_corr) else 0)
            else:
                opt_corrs.append(0)
        
        x = np.arange(len(orig_corrs))
        width = 0.35
        
        ax.bar(x - width/2, orig_corrs, width, label='原始分组', alpha=0.8)
        ax.bar(x + width/2, opt_corrs, width, label='优化分组', alpha=0.8)
        
        ax.set_xlabel('分组', fontsize=12)
        ax.set_ylabel('|相关性|', fontsize=12)
        ax.set_title('优化前后相关性对比', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Group_{i+1}' for i in range(len(orig_corrs))])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_group_plots(self, save_plots=True):
        """创建各分组独立图表"""
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, group in enumerate(self.df['optimized_group'].cat.categories):
            group_data = self.df[self.df['optimized_group'] == group]
            
            if len(group_data) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'优化后分组: {group} (n={len(group_data)})', fontsize=16, fontweight='bold')
                
                # 散点图
                ax1 = axes[0, 0]
                ax1.scatter(group_data[self.ga_col], group_data[self.y_col], 
                           c=colors[i % len(colors)], alpha=0.7, s=60)
                ax1.set_xlabel('孕周 (周)', fontsize=12)
                ax1.set_ylabel('Y染色体浓度 (%)', fontsize=12)
                ax1.set_title(f'{group} - 散点图', fontsize=14)
                ax1.grid(True, alpha=0.3)
                
                # 孕周分布
                ax2 = axes[0, 1]
                ax2.hist(group_data[self.ga_col], bins=20, alpha=0.7, color='skyblue')
                ax2.set_xlabel('孕周 (周)', fontsize=12)
                ax2.set_ylabel('频数', fontsize=12)
                ax2.set_title(f'{group} - 孕周分布', fontsize=14)
                ax2.grid(True, alpha=0.3)
                
                # Y染色体浓度分布
                ax3 = axes[1, 0]
                ax3.hist(group_data[self.y_col], bins=20, alpha=0.7, color='lightcoral')
                ax3.set_xlabel('Y染色体浓度 (%)', fontsize=12)
                ax3.set_ylabel('频数', fontsize=12)
                ax3.set_title(f'{group} - Y染色体浓度分布', fontsize=14)
                ax3.grid(True, alpha=0.3)
                
                # 统计信息
                ax4 = axes[1, 1]
                ax4.axis('off')
                
                corr = group_data[self.ga_col].corr(group_data[self.y_col])
                stats_text = f"{group} 统计信息\n" + "="*20 + "\n\n"
                stats_text += f"样本数: {len(group_data)}\n"
                stats_text += f"BMI范围: {group_data[self.bmi_col].min():.1f} - {group_data[self.bmi_col].max():.1f}\n"
                stats_text += f"孕周范围: {group_data[self.ga_col].min():.1f} - {group_data[self.ga_col].max():.1f}\n"
                stats_text += f"Y染色体范围: {group_data[self.y_col].min():.2f} - {group_data[self.y_col].max():.2f}%\n"
                stats_text += f"相关性: {corr:.3f}\n"
                
                ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace')
                
                plt.tight_layout()
                
                if save_plots:
                    filename = f'optimized_group_{group}_analysis.png'
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"分组图表已保存: {filename}")
                
                plt.close()
    
    def _create_optimization_history_plot(self, save_plots=True):
        """创建优化历史图表"""
        if not self.optimization_history:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        scores = [h['score'] for h in self.optimization_history]
        ax.plot(scores, 'b-', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=self.best_score, color='r', linestyle='--', alpha=0.7, 
                  label=f'最佳得分: {self.best_score:.4f}')
        
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('平均相关性', fontsize=12)
        ax.set_title('优化过程收敛曲线', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig('optimization_history.png', dpi=300, bbox_inches='tight')
            print("优化历史图表已保存: optimization_history.png")
        
        plt.show()
    
    def export_results(self, filename='bmi_optimization_results.csv'):
        """导出优化结果"""
        if self.best_thresholds is None:
            print("请先运行优化")
            return
        
        # 添加优化后的分组信息
        export_df = self.df.copy()
        export_df['optimized_group'] = export_df['optimized_group'].astype(str)
        
        # 保存结果
        export_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"优化结果已保存: {filename}")
        
        # 保存优化报告
        report_filename = 'bmi_optimization_report.txt'
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("BMI分组优化报告\n")
            f.write("="*50 + "\n\n")
            f.write(f"优化时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"原始数据样本数: {len(self.df)}\n")
            f.write(f"最佳阈值: {self.best_thresholds}\n")
            f.write(f"最佳得分: {self.best_score:.4f}\n\n")
            
            f.write("各分组统计:\n")
            f.write("-"*30 + "\n")
            
            for group in self.df['optimized_group'].cat.categories:
                group_data = self.df[self.df['optimized_group'] == group]
                if len(group_data) > 0:
                    corr = group_data[self.ga_col].corr(group_data[self.y_col])
                    f.write(f"\n{group}:\n")
                    f.write(f"  样本数: {len(group_data)}\n")
                    f.write(f"  BMI范围: {group_data[self.bmi_col].min():.2f} - {group_data[self.bmi_col].max():.2f}\n")
                    f.write(f"  相关性: {corr:.3f}\n")
        
        print(f"优化报告已保存: {report_filename}")

def main():
    """主函数"""
    print("BMI分组自动优化工具")
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
    
    # 创建优化器
    optimizer = BMIOptimizer(df, bmi_col, y_col, ga_col)
    
    # 运行优化
    print("\n开始优化过程...")
    
    # 首先尝试网格搜索
    print("1. 网格搜索优化...")
    optimizer.grid_search_optimization()
    
    # 然后使用梯度优化
    print("\n2. 梯度优化...")
    optimizer.optimize_thresholds(initial_thresholds=optimizer.best_thresholds)
    
    # 创建优化后的图表
    print("\n3. 创建优化图表...")
    optimizer.create_optimized_plots()
    
    # 导出结果
    print("\n4. 导出结果...")
    optimizer.export_results()
    
    print("\n优化完成！")
    print(f"最佳BMI分组阈值: {optimizer.best_thresholds}")
    print(f"最佳平均相关性: {optimizer.best_score:.4f}")

if __name__ == "__main__":
    main()
