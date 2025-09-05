#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1 分析代码 v1.3 增强版 - 加入聚类和降维方法
基于清洗后的男性胎儿数据，使用多种高级拟合算法、聚类分析和降维方法
包括：K-means聚类、PCA、t-SNE、集成学习、特征工程等
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score, calinski_harabasz_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体 - 优化版本
import matplotlib
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

# 设置matplotlib后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True

def load_cleaned_data():
    """加载清洗后的数据"""
    print("=== 加载清洗后的数据 ===")
    
    try:
        # 使用路径工具函数加载数据
        from path_utils import load_data_file
        df = load_data_file('male_fetal_data_cleaned.csv', 'csv')
        print(f"成功加载清洗后的数据，形状: {df.shape}")
        
        # 显示基本信息
        print(f"样本数: {len(df)}")
        print(f"列数: {df.shape[1]}")
        
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def explore_cleaned_data(df):
    """探索清洗后的数据"""
    print("\n=== 清洗后数据探索 ===")
    
    # 关键变量统计
    key_vars = ['检测孕周', '孕妇BMI', 'Y染色体浓度', '年龄', '身高', '体重']
    
    print("关键变量统计:")
    for var in key_vars:
        if var in df.columns:
            print(f"\n{var}:")
            print(f"  均值: {df[var].mean():.3f}")
            print(f"  标准差: {df[var].std():.3f}")
            print(f"  最小值: {df[var].min():.3f}")
            print(f"  最大值: {df[var].max():.3f}")
            print(f"  中位数: {df[var].median():.3f}")
    
    # 缺失值统计
    print("\n缺失值统计:")
    missing_stats = df[key_vars].isnull().sum()
    for var, missing_count in missing_stats.items():
        if missing_count > 0:
            print(f"  {var}: {missing_count}个缺失值")
        else:
            print(f"  {var}: 无缺失值")
    
    return df

def feature_engineering(df):
    """特征工程"""
    print("\n=== 特征工程 ===")
    
    # 创建新特征
    df_eng = df.copy()
    
    # 1. 交互特征
    df_eng['孕周_BMI'] = df_eng['检测孕周'] * df_eng['孕妇BMI']
    df_eng['孕周_年龄'] = df_eng['检测孕周'] * df_eng['年龄']
    df_eng['BMI_年龄'] = df_eng['孕妇BMI'] * df_eng['年龄']
    
    # 2. 多项式特征
    df_eng['孕周_平方'] = df_eng['检测孕周'] ** 2
    df_eng['BMI_平方'] = df_eng['孕妇BMI'] ** 2
    df_eng['年龄_平方'] = df_eng['年龄'] ** 2
    
    # 3. 比率特征
    df_eng['BMI_孕周比'] = df_eng['孕妇BMI'] / df_eng['检测孕周']
    df_eng['年龄_孕周比'] = df_eng['年龄'] / df_eng['检测孕周']
    
    # 4. 分类特征
    df_eng['BMI分组'] = pd.cut(df_eng['孕妇BMI'], 
                              bins=[0, 18.5, 25, 30, 35, 100], 
                              labels=['偏瘦', '正常', '超重', '肥胖I级', '肥胖II级'])
    
    df_eng['孕周分组'] = pd.cut(df_eng['检测孕周'], 
                              bins=[0, 12, 16, 20, 25, 30], 
                              labels=['早期', '早中期', '中期', '中晚期', '晚期'])
    
    df_eng['年龄分组'] = pd.cut(df_eng['年龄'], 
                              bins=[0, 25, 30, 35, 100], 
                              labels=['年轻', '中年', '中老年', '老年'])
    
    # 5. 编码分类特征
    le_bmi = LabelEncoder()
    le_ga = LabelEncoder()
    le_age = LabelEncoder()
    
    df_eng['BMI分组_编码'] = le_bmi.fit_transform(df_eng['BMI分组'].astype(str))
    df_eng['孕周分组_编码'] = le_ga.fit_transform(df_eng['孕周分组'].astype(str))
    df_eng['年龄分组_编码'] = le_age.fit_transform(df_eng['年龄分组'].astype(str))
    
    print(f"特征工程完成，新增特征数: {df_eng.shape[1] - df.shape[1]}")
    print(f"总特征数: {df_eng.shape[1]}")
    
    return df_eng, {'le_bmi': le_bmi, 'le_ga': le_ga, 'le_age': le_age}

def clustering_analysis(df, feature_cols):
    """聚类分析"""
    print("\n=== 聚类分析 ===")
    
    # 准备数据
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # 1. K-means聚类
    print("\n--- K-means聚类 ---")
    kmeans_results = {}
    
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # 计算聚类指标
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
        
        # 计算每个聚类的Y染色体浓度统计
        df_temp = df.copy()
        df_temp['cluster'] = cluster_labels
        
        cluster_stats = df_temp.groupby('cluster')['Y染色体浓度'].agg(['count', 'mean', 'std']).round(3)
        
        print(f"  {n_clusters}个聚类: 轮廓系数={silhouette_avg:.3f}, CH指数={calinski_harabasz:.3f}")
        
        kmeans_results[n_clusters] = {
            'model': kmeans,
            'labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'cluster_stats': cluster_stats
        }
    
    # 选择最佳K值
    best_k = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette_score'])
    print(f"最佳K值: {best_k} (轮廓系数: {kmeans_results[best_k]['silhouette_score']:.3f})")
    
    results['kmeans'] = kmeans_results
    results['best_k'] = best_k
    
    # 2. DBSCAN聚类
    print("\n--- DBSCAN聚类 ---")
    dbscan_results = {}
    
    eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    min_samples_values = [5, 10, 15, 20]
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(X_scaled)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            if n_clusters > 1:
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                print(f"  eps={eps}, min_samples={min_samples}: {n_clusters}个聚类, {n_noise}个噪声点, 轮廓系数={silhouette_avg:.3f}")
                
                dbscan_results[f"eps_{eps}_min_{min_samples}"] = {
                    'model': dbscan,
                    'labels': cluster_labels,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette_score': silhouette_avg
                }
    
    results['dbscan'] = dbscan_results
    
    # 3. 层次聚类
    print("\n--- 层次聚类 ---")
    hierarchical_results = {}
    
    for n_clusters in range(2, 8):
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = hierarchical.fit_predict(X_scaled)
        
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
        
        print(f"  {n_clusters}个聚类: 轮廓系数={silhouette_avg:.3f}, CH指数={calinski_harabasz:.3f}")
        
        hierarchical_results[n_clusters] = {
            'model': hierarchical,
            'labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz
        }
    
    results['hierarchical'] = hierarchical_results
    
    return results, X_scaled, scaler

def dimensionality_reduction_analysis(X_scaled):
    """降维分析"""
    print("\n=== 降维分析 ===")
    
    results = {}
    
    # 1. PCA分析
    print("\n--- PCA分析 ---")
    pca_results = {}
    
    for n_components in range(1, min(6, X_scaled.shape[1] + 1)):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"  {n_components}个主成分: 解释方差比={explained_variance_ratio}, 累积方差比={cumulative_variance[-1]:.3f}")
        
        pca_results[n_components] = {
            'model': pca,
            'transformed_data': X_pca,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance
        }
    
    results['pca'] = pca_results
    
    # 2. t-SNE分析
    print("\n--- t-SNE分析 ---")
    tsne_results = {}
    
    perplexities = [5, 10, 20, 30, 50]
    
    for perplexity in perplexities:
        if perplexity < X_scaled.shape[0] / 3:  # t-SNE要求perplexity < n_samples / 3
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X_scaled)
            
            print(f"  perplexity={perplexity}: 降维到2D完成")
            
            tsne_results[perplexity] = {
                'model': tsne,
                'transformed_data': X_tsne
            }
    
    results['tsne'] = tsne_results
    
    return results

def feature_selection_analysis(df, feature_cols, target_col='Y染色体浓度'):
    """特征选择分析"""
    print("\n=== 特征选择分析 ===")
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    results = {}
    
    # 1. 单变量特征选择
    print("\n--- 单变量特征选择 ---")
    
    # F统计量
    f_selector = SelectKBest(score_func=f_regression, k='all')
    f_scores = f_selector.fit(X, y)
    
    # 互信息
    mi_selector = SelectKBest(score_func=mutual_info_regression, k='all')
    mi_scores = mi_selector.fit(X, y)
    
    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'f_score': f_scores.scores_,
        'f_pvalue': f_scores.pvalues_,
        'mi_score': mi_scores.scores_
    })
    
    feature_importance = feature_importance.sort_values('f_score', ascending=False)
    
    print("特征重要性排序 (F统计量):")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: F={row['f_score']:.3f}, p={row['f_pvalue']:.6f}, MI={row['mi_score']:.3f}")
    
    results['feature_importance'] = feature_importance
    
    # 2. 递归特征消除
    print("\n--- 递归特征消除 ---")
    from sklearn.feature_selection import RFE
    
    # 使用随机森林作为基础模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=5)
    rfe.fit(X, y)
    
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
    print(f"RFE选择的5个最重要特征: {selected_features}")
    
    results['rfe'] = {
        'selector': rfe,
        'selected_features': selected_features,
        'feature_ranking': rfe.ranking_
    }
    
    return results

def ensemble_learning_analysis(df, feature_cols, target_col='Y染色体浓度'):
    """集成学习分析"""
    print("\n=== 集成学习分析 ===")
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # 1. 投票回归器
    print("\n--- 投票回归器 ---")
    
    # 定义基础模型
    models = [
        ('ridge', Ridge(alpha=1.0)),
        ('lasso', Lasso(alpha=0.1)),
        ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]
    
    # 硬投票
    voting_hard = VotingRegressor(models)
    cv_scores_hard = cross_val_score(voting_hard, X_scaled, y, cv=5, scoring='r2')
    voting_hard.fit(X_scaled, y)
    y_pred_hard = voting_hard.predict(X_scaled)
    
    r2_hard = r2_score(y, y_pred_hard)
    rmse_hard = np.sqrt(mean_squared_error(y, y_pred_hard))
    
    print(f"硬投票: R²={r2_hard:.4f}, RMSE={rmse_hard:.4f}, 交叉验证R²={cv_scores_hard.mean():.4f}±{cv_scores_hard.std():.4f}")
    
    # 软投票
    voting_soft = VotingRegressor(models, weights=[1, 1, 1, 2, 2])  # 给树模型更高权重
    cv_scores_soft = cross_val_score(voting_soft, X_scaled, y, cv=5, scoring='r2')
    voting_soft.fit(X_scaled, y)
    y_pred_soft = voting_soft.predict(X_scaled)
    
    r2_soft = r2_score(y, y_pred_soft)
    rmse_soft = np.sqrt(mean_squared_error(y, y_pred_soft))
    
    print(f"软投票: R²={r2_soft:.4f}, RMSE={rmse_soft:.4f}, 交叉验证R²={cv_scores_soft.mean():.4f}±{cv_scores_soft.std():.4f}")
    
    results['voting'] = {
        'hard': {'model': voting_hard, 'r2': r2_hard, 'rmse': rmse_hard, 'cv_r2': cv_scores_hard.mean()},
        'soft': {'model': voting_soft, 'r2': r2_soft, 'rmse': rmse_soft, 'cv_r2': cv_scores_soft.mean()}
    }
    
    # 2. Bagging回归器
    print("\n--- Bagging回归器 ---")
    
    bagging_models = {
        'Bagging_RF': BaggingRegressor(estimator=RandomForestRegressor(n_estimators=50), n_estimators=10, random_state=42),
        'Bagging_DT': BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10, random_state=42),
        'Bagging_SVR': BaggingRegressor(estimator=SVR(), n_estimators=10, random_state=42)
    }
    
    for name, model in bagging_models.items():
        try:
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            print(f"{name}: R²={r2:.4f}, RMSE={rmse:.4f}, 交叉验证R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
            
            results[name] = {
                'model': model,
                'r2': r2,
                'rmse': rmse,
                'cv_r2': cv_scores.mean(),
                'y_pred': y_pred
            }
        except Exception as e:
            print(f"{name}训练失败: {e}")
    
    return results, X_scaled, scaler

def create_comprehensive_visualizations(df, clustering_results, dim_reduction_results, feature_selection_results, ensemble_results, df_eng=None, feature_cols=None):
    """创建综合可视化图表"""
    print("\n=== 创建综合可视化图表 ===")
    
    # 设置图表样式和字体
    plt.style.use('default')
    
    # 尝试设置中文字体
    try:
        from matplotlib.font_manager import FontProperties
        import matplotlib.font_manager as fm
        
        chinese_fonts = []
        for font in fm.fontManager.ttflist:
            if 'SimHei' in font.name or 'Microsoft YaHei' in font.name or 'WenQuanYi' in font.name:
                chinese_fonts.append(font.name)
        
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = chinese_fonts[:3] + ['DejaVu Sans']
            print(f"使用中文字体: {chinese_fonts[:3]}")
        else:
            print("未找到中文字体，使用默认字体")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            
    except Exception as e:
        print(f"字体设置警告: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 设置图表参数
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['font.size'] = 10
    
    # 创建大图
    fig = plt.figure(figsize=(24, 20))
    
    # 1. 原始数据散点图
    plt.subplot(4, 4, 1)
    plt.scatter(df['检测孕周'], df['Y染色体浓度'], alpha=0.6, s=20, color='blue')
    plt.xlabel('检测孕周 (周)', fontsize=12, fontweight='bold')
    plt.ylabel('Y染色体浓度 (%)', fontsize=12, fontweight='bold')
    plt.title('Y染色体浓度与孕周关系', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # 2. BMI散点图
    plt.subplot(4, 4, 2)
    plt.scatter(df['孕妇BMI'], df['Y染色体浓度'], alpha=0.6, s=20, color='orange')
    plt.xlabel('孕妇BMI (kg/m²)', fontsize=12, fontweight='bold')
    plt.ylabel('Y染色体浓度 (%)', fontsize=12, fontweight='bold')
    plt.title('Y染色体浓度与BMI关系', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # 3. 年龄散点图
    plt.subplot(4, 4, 3)
    plt.scatter(df['年龄'], df['Y染色体浓度'], alpha=0.6, s=20, color='green')
    plt.xlabel('年龄 (岁)', fontsize=12, fontweight='bold')
    plt.ylabel('Y染色体浓度 (%)', fontsize=12, fontweight='bold')
    plt.title('Y染色体浓度与年龄关系', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # 4. K-means聚类结果
    if 'kmeans' in clustering_results and 'best_k' in clustering_results:
        plt.subplot(4, 4, 4)
        best_k = clustering_results['best_k']
        kmeans_result = clustering_results['kmeans'][best_k]
        
        # 使用前两个主成分进行可视化
        if 'pca' in dim_reduction_results and 2 in dim_reduction_results['pca']:
            pca_data = dim_reduction_results['pca'][2]['transformed_data']
            scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_result['labels'], 
                                cmap='viridis', alpha=0.6, s=20)
            plt.colorbar(scatter, label='聚类标签')
            plt.xlabel('第一主成分', fontsize=12, fontweight='bold')
            plt.ylabel('第二主成分', fontsize=12, fontweight='bold')
            plt.title(f'K-means聚类结果 (K={best_k})', fontsize=14, fontweight='bold', pad=20)
        else:
            plt.scatter(df['检测孕周'], df['Y染色体浓度'], c=kmeans_result['labels'], 
                       cmap='viridis', alpha=0.6, s=20)
            plt.xlabel('检测孕周 (周)', fontsize=12, fontweight='bold')
            plt.ylabel('Y染色体浓度 (%)', fontsize=12, fontweight='bold')
            plt.title(f'K-means聚类结果 (K={best_k})', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
    
    # 5. PCA降维结果
    if 'pca' in dim_reduction_results and 2 in dim_reduction_results['pca']:
        plt.subplot(4, 4, 5)
        pca_data = dim_reduction_results['pca'][2]['transformed_data']
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['Y染色体浓度'], 
                   cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(label='Y染色体浓度 (%)')
        plt.xlabel('第一主成分', fontsize=12, fontweight='bold')
        plt.ylabel('第二主成分', fontsize=12, fontweight='bold')
        plt.title('PCA降维结果', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
    
    # 6. t-SNE降维结果
    if 'tsne' in dim_reduction_results and 20 in dim_reduction_results['tsne']:
        plt.subplot(4, 4, 6)
        tsne_data = dim_reduction_results['tsne'][20]['transformed_data']
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=df['Y染色体浓度'], 
                   cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(label='Y染色体浓度 (%)')
        plt.xlabel('t-SNE 1', fontsize=12, fontweight='bold')
        plt.ylabel('t-SNE 2', fontsize=12, fontweight='bold')
        plt.title('t-SNE降维结果', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
    
    # 7. 特征重要性
    if 'feature_importance' in feature_selection_results:
        plt.subplot(4, 4, 7)
        feature_importance = feature_selection_results['feature_importance'].head(10)
        
        bars = plt.barh(range(len(feature_importance)), feature_importance['f_score'], 
                       color='lightblue')
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('F统计量', fontsize=12, fontweight='bold')
        plt.title('特征重要性 (F统计量)', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, feature_importance['f_score'])):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{score:.2f}', ha='left', va='center', fontsize=8)
    
    # 8. 聚类轮廓系数
    if 'kmeans' in clustering_results:
        plt.subplot(4, 4, 8)
        k_values = list(clustering_results['kmeans'].keys())
        silhouette_scores = [clustering_results['kmeans'][k]['silhouette_score'] for k in k_values]
        
        plt.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('聚类数 K', fontsize=12, fontweight='bold')
        plt.ylabel('轮廓系数', fontsize=12, fontweight='bold')
        plt.title('K-means聚类轮廓系数', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        
        # 标记最佳K值
        if 'best_k' in clustering_results:
            best_k = clustering_results['best_k']
            best_score = clustering_results['kmeans'][best_k]['silhouette_score']
            plt.plot(best_k, best_score, 'ro', markersize=12, label=f'最佳K={best_k}')
            plt.legend()
    
    # 9. PCA解释方差比
    if 'pca' in dim_reduction_results:
        plt.subplot(4, 4, 9)
        pca_results = dim_reduction_results['pca']
        n_components = list(pca_results.keys())
        explained_variance = [pca_results[n]['cumulative_variance'][-1] for n in n_components]
        
        plt.plot(n_components, explained_variance, 'go-', linewidth=2, markersize=8)
        plt.xlabel('主成分数', fontsize=12, fontweight='bold')
        plt.ylabel('累积解释方差比', fontsize=12, fontweight='bold')
        plt.title('PCA解释方差比', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
    
    # 10. 集成学习性能对比
    if 'voting' in ensemble_results:
        plt.subplot(4, 4, 10)
        methods = ['硬投票', '软投票']
        r2_scores = [ensemble_results['voting']['hard']['r2'], 
                    ensemble_results['voting']['soft']['r2']]
        
        bars = plt.bar(methods, r2_scores, color=['lightblue', 'lightgreen'])
        plt.ylabel('R²', fontsize=12, fontweight='bold')
        plt.title('集成学习性能对比', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 11. 聚类结果统计
    if 'kmeans' in clustering_results and 'best_k' in clustering_results:
        plt.subplot(4, 4, 11)
        best_k = clustering_results['best_k']
        cluster_stats = clustering_results['kmeans'][best_k]['cluster_stats']
        
        cluster_ids = cluster_stats.index
        cluster_means = cluster_stats['mean']
        
        bars = plt.bar(cluster_ids, cluster_means, color='lightcoral')
        plt.xlabel('聚类ID', fontsize=12, fontweight='bold')
        plt.ylabel('平均Y染色体浓度 (%)', fontsize=12, fontweight='bold')
        plt.title(f'各聚类Y染色体浓度均值 (K={best_k})', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, mean_val in zip(bars, cluster_means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 12. 残差分析
    if 'voting' in ensemble_results:
        plt.subplot(4, 4, 12)
        
        # 使用正确的特征进行预测
        if df_eng is not None and feature_cols is not None:
            # 使用特征工程后的数据
            X_ensemble = df_eng[feature_cols].values
            scaler_ensemble = StandardScaler()
            X_ensemble_scaled = scaler_ensemble.fit_transform(X_ensemble)
            y_pred = ensemble_results['voting']['soft']['model'].predict(X_ensemble_scaled)
        else:
            # 如果只有原始数据，使用原始特征
            X_ensemble = df[['检测孕周', '孕妇BMI', '年龄']].values
            scaler_ensemble = StandardScaler()
            X_ensemble_scaled = scaler_ensemble.fit_transform(X_ensemble)
            y_pred = ensemble_results['voting']['soft']['model'].predict(X_ensemble_scaled)
        
        residuals = df['Y染色体浓度'] - y_pred
        
        plt.scatter(y_pred, residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('预测值', fontsize=12, fontweight='bold')
        plt.ylabel('残差', fontsize=12, fontweight='bold')
        plt.title('集成学习残差图', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
    
    # 调整布局和保存
    plt.tight_layout(pad=3.0)
    
    # 保存图表
    try:
        plt.savefig('t1_analysis_enhanced_v1.3.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print("增强版可视化图表已保存为: t1_analysis_enhanced_v1.3.png")
    except Exception as e:
        print(f"保存图表时出错: {e}")
        plt.savefig('t1_analysis_enhanced_v1.3.png', dpi=150, bbox_inches='tight')
    
    # 显示图表
    try:
        plt.show()
    except Exception as e:
        print(f"显示图表时出错: {e}")
        print("图表已保存，但无法在终端中显示")
    
    return fig

def generate_enhanced_report(df, clustering_results, dim_reduction_results, feature_selection_results, ensemble_results):
    """生成增强版报告"""
    print("\n=== 生成增强版报告 ===")
    
    report = []
    report.append("# T1分析增强版报告 v1.3 - 聚类与降维分析")
    report.append("=" * 60)
    report.append("")
    
    # 数据概况
    report.append("## 数据概况")
    report.append(f"- 样本数: {len(df)}")
    report.append(f"- 特征数: {df.shape[1]}")
    report.append(f"- 目标变量: Y染色体浓度")
    report.append("")
    
    # 聚类分析结果
    if 'kmeans' in clustering_results:
        report.append("## 聚类分析结果")
        report.append("")
        
        best_k = clustering_results['best_k']
        kmeans_result = clustering_results['kmeans'][best_k]
        
        report.append(f"### K-means聚类")
        report.append(f"- 最佳聚类数: {best_k}")
        report.append(f"- 轮廓系数: {kmeans_result['silhouette_score']:.3f}")
        report.append(f"- CH指数: {kmeans_result['calinski_harabasz_score']:.3f}")
        report.append("")
        
        report.append("各聚类统计:")
        cluster_stats = kmeans_result['cluster_stats']
        for cluster_id, stats in cluster_stats.iterrows():
            report.append(f"- 聚类{cluster_id}: 样本数={stats['count']}, 均值={stats['mean']:.3f}%, 标准差={stats['std']:.3f}%")
        report.append("")
    
    # 降维分析结果
    if 'pca' in dim_reduction_results:
        report.append("## 降维分析结果")
        report.append("")
        
        report.append("### PCA分析")
        for n_components, result in dim_reduction_results['pca'].items():
            cumulative_var = result['cumulative_variance'][-1]
            report.append(f"- {n_components}个主成分: 累积解释方差比={cumulative_var:.3f}")
        report.append("")
    
    # 特征选择结果
    if 'feature_importance' in feature_selection_results:
        report.append("## 特征选择结果")
        report.append("")
        
        report.append("### 特征重要性排序 (前10名)")
        top_features = feature_selection_results['feature_importance'].head(10)
        for i, row in top_features.iterrows():
            report.append(f"- {row['feature']}: F={row['f_score']:.3f}, p={row['f_pvalue']:.6f}, MI={row['mi_score']:.3f}")
        report.append("")
    
    # 集成学习结果
    if 'voting' in ensemble_results:
        report.append("## 集成学习结果")
        report.append("")
        
        report.append("### 投票回归器")
        hard_result = ensemble_results['voting']['hard']
        soft_result = ensemble_results['voting']['soft']
        
        report.append(f"- 硬投票: R²={hard_result['r2']:.4f}, RMSE={hard_result['rmse']:.4f}, 交叉验证R²={hard_result['cv_r2']:.4f}")
        report.append(f"- 软投票: R²={soft_result['r2']:.4f}, RMSE={soft_result['rmse']:.4f}, 交叉验证R²={soft_result['cv_r2']:.4f}")
        report.append("")
    
    # 建议
    report.append("## 建议")
    report.append("1. 根据聚类结果分析不同群体的特征")
    report.append("2. 使用PCA降维减少特征维度")
    report.append("3. 基于特征重要性进行特征选择")
    report.append("4. 使用集成学习提高预测精度")
    report.append("5. 结合领域知识解释聚类结果")
    report.append("")
    
    # 保存报告
    report_text = "\n".join(report)
    with open('t1_analysis_enhanced_report_v1.3.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("增强版报告已保存为: t1_analysis_enhanced_report_v1.3.md")
    print("\n报告内容:")
    print(report_text)
    
    return report_text

def main():
    """主函数"""
    print("T1 分析代码 v1.3 增强版 - 聚类与降维分析")
    print("基于清洗后的男性胎儿数据，使用多种高级拟合算法、聚类分析和降维方法")
    print("=" * 80)
    
    # 1. 加载数据
    df = load_cleaned_data()
    if df is None:
        return
    
    # 2. 数据探索
    df = explore_cleaned_data(df)
    
    # 3. 特征工程
    df_eng, encoders = feature_engineering(df)
    
    # 4. 选择特征
    feature_cols = ['检测孕周', '孕妇BMI', '年龄', '孕周_BMI', '孕周_年龄', 'BMI_年龄', 
                   '孕周_平方', 'BMI_平方', '年龄_平方', 'BMI_孕周比', '年龄_孕周比',
                   'BMI分组_编码', '孕周分组_编码', '年龄分组_编码']
    
    # 5. 聚类分析
    clustering_results, X_scaled, scaler = clustering_analysis(df_eng, feature_cols)
    
    # 6. 降维分析
    dim_reduction_results = dimensionality_reduction_analysis(X_scaled)
    
    # 7. 特征选择分析
    feature_selection_results = feature_selection_analysis(df_eng, feature_cols)
    
    # 8. 集成学习分析
    ensemble_results, X_scaled_ensemble, scaler_ensemble = ensemble_learning_analysis(df_eng, feature_cols)
    
    # 9. 创建综合可视化
    create_comprehensive_visualizations(df, clustering_results, dim_reduction_results, 
                                      feature_selection_results, ensemble_results, 
                                      df_eng, feature_cols)
    
    # 10. 生成增强版报告
    generate_enhanced_report(df, clustering_results, dim_reduction_results, 
                           feature_selection_results, ensemble_results)
    
    print("\n=== 增强版分析完成 ===")
    print("所有结果已保存到当前目录")

if __name__ == "__main__":
    main()
