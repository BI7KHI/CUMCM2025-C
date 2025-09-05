#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1 分析代码 v1.3 增强版 - 测试版本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """主函数"""
    print("T1 分析代码 v1.3 增强版 - 测试版本")
    print("=" * 50)
    
    # 加载数据
    try:
        from path_utils import load_data_file
        df = load_data_file('male_fetal_data_cleaned.csv', 'csv')
        print(f"成功加载数据，形状: {df.shape}")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 特征工程
    print("\n=== 特征工程 ===")
    df['孕周_BMI'] = df['检测孕周'] * df['孕妇BMI']
    df['孕周_年龄'] = df['检测孕周'] * df['年龄']
    df['BMI_年龄'] = df['孕妇BMI'] * df['年龄']
    df['孕周_平方'] = df['检测孕周'] ** 2
    df['BMI_平方'] = df['孕妇BMI'] ** 2
    df['年龄_平方'] = df['年龄'] ** 2
    
    feature_cols = ['检测孕周', '孕妇BMI', '年龄', '孕周_BMI', '孕周_年龄', 'BMI_年龄', 
                   '孕周_平方', 'BMI_平方', '年龄_平方']
    
    print(f"新增特征数: {len(feature_cols) - 3}")
    print(f"总特征数: {len(feature_cols)}")
    
    # 准备数据
    X = df[feature_cols].values
    y = df['Y染色体浓度'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means聚类
    print("\n=== K-means聚类分析 ===")
    kmeans_results = {}
    
    for n_clusters in range(2, 8):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # 计算聚类指标
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        # 计算每个聚类的Y染色体浓度统计
        df_temp = df.copy()
        df_temp['cluster'] = cluster_labels
        cluster_stats = df_temp.groupby('cluster')['Y染色体浓度'].agg(['count', 'mean', 'std']).round(3)
        
        print(f"  {n_clusters}个聚类: 轮廓系数={silhouette_avg:.3f}")
        print(f"    各聚类Y染色体浓度均值: {cluster_stats['mean'].values}")
        
        kmeans_results[n_clusters] = {
            'model': kmeans,
            'labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'cluster_stats': cluster_stats
        }
    
    # 选择最佳K值
    best_k = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette_score'])
    print(f"最佳K值: {best_k} (轮廓系数: {kmeans_results[best_k]['silhouette_score']:.3f})")
    
    # PCA降维
    print("\n=== PCA降维分析 ===")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"前2个主成分解释方差比: {pca.explained_variance_ratio_}")
    print(f"累积解释方差比: {pca.explained_variance_ratio_.sum():.3f}")
    
    # 集成学习
    print("\n=== 集成学习分析 ===")
    
    # 定义基础模型
    models = [
        ('ridge', Ridge(alpha=1.0)),
        ('lasso', Lasso(alpha=0.1)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ]
    
    # 投票回归器
    voting = VotingRegressor(models)
    
    # 交叉验证
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(voting, X_scaled, y, cv=5, scoring='r2')
    
    # 训练模型
    voting.fit(X_scaled, y)
    y_pred = voting.predict(X_scaled)
    
    # 评估指标
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"投票回归器: R²={r2:.4f}, RMSE={rmse:.4f}")
    print(f"交叉验证R²: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    # 创建可视化
    print("\n=== 创建可视化 ===")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 原始数据散点图
    axes[0, 0].scatter(df['检测孕周'], df['Y染色体浓度'], alpha=0.6, s=20, color='blue')
    axes[0, 0].set_xlabel('检测孕周 (周)')
    axes[0, 0].set_ylabel('Y染色体浓度 (%)')
    axes[0, 0].set_title('Y染色体浓度与孕周关系')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. BMI散点图
    axes[0, 1].scatter(df['孕妇BMI'], df['Y染色体浓度'], alpha=0.6, s=20, color='orange')
    axes[0, 1].set_xlabel('孕妇BMI (kg/m²)')
    axes[0, 1].set_ylabel('Y染色体浓度 (%)')
    axes[0, 1].set_title('Y染色体浓度与BMI关系')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 年龄散点图
    axes[0, 2].scatter(df['年龄'], df['Y染色体浓度'], alpha=0.6, s=20, color='green')
    axes[0, 2].set_xlabel('年龄 (岁)')
    axes[0, 2].set_ylabel('Y染色体浓度 (%)')
    axes[0, 2].set_title('Y染色体浓度与年龄关系')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. K-means聚类结果
    best_kmeans = kmeans_results[best_k]
    scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=best_kmeans['labels'], 
                                cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=axes[1, 0], label='聚类标签')
    axes[1, 0].set_xlabel('第一主成分')
    axes[1, 0].set_ylabel('第二主成分')
    axes[1, 0].set_title(f'K-means聚类结果 (K={best_k})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. PCA降维结果
    scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=df['Y染色体浓度'], 
                                cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=axes[1, 1], label='Y染色体浓度 (%)')
    axes[1, 1].set_xlabel('第一主成分')
    axes[1, 1].set_ylabel('第二主成分')
    axes[1, 1].set_title('PCA降维结果')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 预测 vs 实际
    axes[1, 2].scatter(df['Y染色体浓度'], y_pred, alpha=0.6, s=20)
    axes[1, 2].plot([df['Y染色体浓度'].min(), df['Y染色体浓度'].max()], 
                    [df['Y染色体浓度'].min(), df['Y染色体浓度'].max()], 'r--', linewidth=2)
    axes[1, 2].set_xlabel('实际Y染色体浓度 (%)')
    axes[1, 2].set_ylabel('预测Y染色体浓度 (%)')
    axes[1, 2].set_title('预测 vs 实际')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_enhanced_visualization_v1.3.png', dpi=300, bbox_inches='tight')
    print("增强版测试可视化图表已保存为: test_enhanced_visualization_v1.3.png")
    
    # 生成测试报告
    report = f"""# T1分析增强版测试报告 v1.3

## 数据概况
- 样本数: {len(df)}
- 原始特征数: 3 (检测孕周, 孕妇BMI, 年龄)
- 工程特征数: {len(feature_cols) - 3}
- 总特征数: {len(feature_cols)}

## 聚类分析结果
- 最佳聚类数: {best_k}
- 轮廓系数: {kmeans_results[best_k]['silhouette_score']:.3f}

### 各聚类统计
"""
    
    cluster_stats = kmeans_results[best_k]['cluster_stats']
    for cluster_id, stats in cluster_stats.iterrows():
        report += f"- 聚类{cluster_id}: 样本数={stats['count']}, 均值={stats['mean']:.3f}%, 标准差={stats['std']:.3f}%\n"
    
    report += f"""
## PCA降维结果
- 前2个主成分解释方差比: {pca.explained_variance_ratio_}
- 累积解释方差比: {pca.explained_variance_ratio_.sum():.3f}

## 集成学习结果
- 投票回归器R²: {r2:.4f}
- 投票回归器RMSE: {rmse:.4f}
- 交叉验证R²: {cv_scores.mean():.4f}±{cv_scores.std():.4f}

## 结论
增强版v1.3成功实现了聚类分析、降维分析和集成学习，为后续的模型优化提供了更多选择。
"""
    
    with open('test_enhanced_report_v1.3.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("增强版测试报告已保存为: test_enhanced_report_v1.3.md")
    print("\n=== 增强版测试完成 ===")

if __name__ == "__main__":
    main()

