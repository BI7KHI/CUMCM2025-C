#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一算法评估系统 v1.3
整合所有机器学习算法，提供全面的模型评估和拟合结果显示
包括：线性回归、多项式回归、样条回归、机器学习模型、集成学习、聚类分析、降维分析
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
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体 - 优化版本
import matplotlib
import matplotlib.font_manager as fm

# 尝试设置中文字体
try:
    # 查找可用的中文字体
    chinese_fonts = []
    for font in fm.fontManager.ttflist:
        if any(name in font.name for name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi', 'Noto Sans CJK', 'Source Han Sans']):
            chinese_fonts.append(font.name)
    
    if chinese_fonts:
        # 使用找到的中文字体
        font_list = chinese_fonts[:3] + ['DejaVu Sans', 'Arial Unicode MS']
        matplotlib.rcParams['font.sans-serif'] = font_list
        print(f"使用中文字体: {chinese_fonts[:3]}")
    else:
        # 如果没有找到中文字体，使用默认设置
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        print("未找到中文字体，使用默认字体")
        
except Exception as e:
    print(f"字体设置警告: {e}")
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

# 设置matplotlib参数
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.autolayout'] = True

# 设置plt参数
plt.rcParams['font.sans-serif'] = matplotlib.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True

class UnifiedAlgorithmEvaluator:
    """统一算法评估器"""
    
    def __init__(self):
        self.df = None
        self.df_eng = None
        self.feature_cols = None
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
    
    def setup_chinese_font(self):
        """设置中文字体"""
        import matplotlib.font_manager as fm
        
        # 查找系统中可用的中文字体
        chinese_fonts = []
        for font in fm.fontManager.ttflist:
            font_name = font.name.lower()
            if any(name in font_name for name in ['microsoft', 'simhei', 'simsun', 'kaiti', 'fangsong', 'wenquanyi', 'noto', 'source han']):
                chinese_fonts.append(font.name)
        
        # 设置字体优先级
        if chinese_fonts:
            font_priority = chinese_fonts[:3] + ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            print(f"找到中文字体: {chinese_fonts[:3]}")
        else:
            font_priority = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            print("未找到中文字体，使用默认字体")
        
        # 强制设置matplotlib字体
        matplotlib.rcParams['font.sans-serif'] = font_priority
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.size'] = 10
        
        # 设置plt字体
        plt.rcParams['font.sans-serif'] = font_priority
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
        
        # 清除matplotlib字体缓存
        try:
            fm._rebuild()
        except:
            pass
        
        return font_priority
        
    def load_data(self):
        """加载数据"""
        print("=== 加载数据 ===")
        
        try:
            from path_utils import load_data_file
            self.df = load_data_file('male_fetal_data_cleaned.csv', 'csv')
            print(f"成功加载数据，形状: {self.df.shape}")
            print(f"样本数: {len(self.df)}")
            print(f"特征数: {self.df.shape[1]}")
            
            # 检查中文字符列名
            chinese_cols = [col for col in self.df.columns if any('\u4e00' <= char <= '\u9fff' for char in str(col))]
            if chinese_cols:
                print(f"检测到中文字段: {chinese_cols[:5]}...")  # 只显示前5个
            
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def explore_data(self):
        """数据探索"""
        print("\n=== 数据探索 ===")
        
        key_vars = ['检测孕周', '孕妇BMI', 'Y染色体浓度', '年龄', '身高', '体重']
        
        print("关键变量统计:")
        for var in key_vars:
            if var in self.df.columns:
                print(f"\n{var}:")
                print(f"  均值: {self.df[var].mean():.3f}")
                print(f"  标准差: {self.df[var].std():.3f}")
                print(f"  最小值: {self.df[var].min():.3f}")
                print(f"  最大值: {self.df[var].max():.3f}")
                print(f"  中位数: {self.df[var].median():.3f}")
        
        # 缺失值统计
        print("\n缺失值统计:")
        missing_stats = self.df[key_vars].isnull().sum()
        for var, missing_count in missing_stats.items():
            if missing_count > 0:
                print(f"  {var}: {missing_count}个缺失值")
            else:
                print(f"  {var}: 无缺失值")
    
    def feature_engineering(self):
        """特征工程"""
        print("\n=== 特征工程 ===")
        
        self.df_eng = self.df.copy()
        
        # 1. 交互特征
        self.df_eng['孕周_BMI'] = self.df_eng['检测孕周'] * self.df_eng['孕妇BMI']
        self.df_eng['孕周_年龄'] = self.df_eng['检测孕周'] * self.df_eng['年龄']
        self.df_eng['BMI_年龄'] = self.df_eng['孕妇BMI'] * self.df_eng['年龄']
        
        # 2. 多项式特征
        self.df_eng['孕周_平方'] = self.df_eng['检测孕周'] ** 2
        self.df_eng['BMI_平方'] = self.df_eng['孕妇BMI'] ** 2
        self.df_eng['年龄_平方'] = self.df_eng['年龄'] ** 2
        
        # 3. 比率特征
        self.df_eng['BMI_孕周比'] = self.df_eng['孕妇BMI'] / self.df_eng['检测孕周']
        self.df_eng['年龄_孕周比'] = self.df_eng['年龄'] / self.df_eng['检测孕周']
        
        # 4. 分类特征
        self.df_eng['BMI分组'] = pd.cut(self.df_eng['孕妇BMI'], 
                                      bins=[0, 18.5, 25, 30, 35, 100], 
                                      labels=['偏瘦', '正常', '超重', '肥胖I级', '肥胖II级'])
        
        self.df_eng['孕周分组'] = pd.cut(self.df_eng['检测孕周'], 
                                      bins=[0, 12, 16, 20, 25, 30], 
                                      labels=['早期', '早中期', '中期', '中晚期', '晚期'])
        
        self.df_eng['年龄分组'] = pd.cut(self.df_eng['年龄'], 
                                      bins=[0, 25, 30, 35, 100], 
                                      labels=['年轻', '中年', '中老年', '老年'])
        
        # 5. 编码分类特征
        le_bmi = LabelEncoder()
        le_ga = LabelEncoder()
        le_age = LabelEncoder()
        
        self.df_eng['BMI分组_编码'] = le_bmi.fit_transform(self.df_eng['BMI分组'].astype(str))
        self.df_eng['孕周分组_编码'] = le_ga.fit_transform(self.df_eng['孕周分组'].astype(str))
        self.df_eng['年龄分组_编码'] = le_age.fit_transform(self.df_eng['年龄分组'].astype(str))
        
        # 设置特征列
        self.feature_cols = ['检测孕周', '孕妇BMI', '年龄', '孕周_BMI', '孕周_年龄', 'BMI_年龄', 
                           '孕周_平方', 'BMI_平方', '年龄_平方', 'BMI_孕周比', '年龄_孕周比',
                           'BMI分组_编码', '孕周分组_编码', '年龄分组_编码']
        
        print(f"特征工程完成，新增特征数: {len(self.feature_cols) - 3}")
        print(f"总特征数: {len(self.feature_cols)}")
        
        return {'le_bmi': le_bmi, 'le_ga': le_ga, 'le_age': le_age}
    
    def correlation_analysis(self):
        """相关性分析"""
        print("\n=== 相关性分析 ===")
        
        key_vars = ['检测孕周', '孕妇BMI', 'Y染色体浓度', '年龄']
        corr_data = self.df[key_vars].dropna()
        
        print(f"相关性分析样本数: {len(corr_data)}")
        
        # Pearson相关系数
        print("\nPearson相关系数:")
        pearson_corr = corr_data.corr(method='pearson')
        print(pearson_corr.round(4))
        
        # Spearman相关系数
        print("\nSpearman相关系数:")
        spearman_corr = corr_data.corr(method='spearman')
        print(spearman_corr.round(4))
        
        # 与Y染色体浓度的详细相关性
        print("\n与Y染色体浓度的详细相关性:")
        y_col = 'Y染色体浓度'
        
        for var in ['检测孕周', '孕妇BMI', '年龄']:
            if var in corr_data.columns:
                pearson_r, pearson_p = pearsonr(corr_data[var], corr_data[y_col])
                spearman_r, spearman_p = spearmanr(corr_data[var], corr_data[y_col])
                
                print(f"\n{var} vs {y_col}:")
                print(f"  Pearson: r={pearson_r:.4f}, p={pearson_p:.6f}")
                print(f"  Spearman: ρ={spearman_r:.4f}, p={spearman_p:.6f}")
        
        self.results['correlation'] = {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'data': corr_data
        }
    
    def polynomial_regression_analysis(self, max_degree=5):
        """多项式回归分析"""
        print(f"\n=== 多项式回归分析 (最高{max_degree}次) ===")
        
        X = self.df[['检测孕周', '孕妇BMI', '年龄']].values
        y = self.df['Y染色体浓度'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        for degree in range(1, max_degree + 1):
            print(f"\n--- {degree}次多项式回归 ---")
            
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly_features.fit_transform(X_scaled)
            
            model = LinearRegression()
            cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
            
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            print(f"R²: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"交叉验证R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            results[degree] = {
                'model': model,
                'poly_features': poly_features,
                'scaler': scaler,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'y_pred': y_pred
            }
            
            # 更新最佳模型
            if cv_scores.mean() > self.best_score:
                self.best_score = cv_scores.mean()
                self.best_model = f"多项式{degree}次"
        
        self.results['polynomial'] = results
        return results
    
    def spline_regression_analysis(self, n_knots_list=[3, 5, 7, 10]):
        """样条回归分析"""
        print(f"\n=== 样条回归分析 ===")
        
        X = self.df[['检测孕周', '孕妇BMI', '年龄']].values
        y = self.df['Y染色体浓度'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        for n_knots in n_knots_list:
            print(f"\n--- 样条回归 (节点数: {n_knots}) ---")
            
            try:
                spline_transformer = SplineTransformer(n_knots=n_knots, degree=3)
                X_spline = spline_transformer.fit_transform(X_scaled)
                
                model = LinearRegression()
                cv_scores = cross_val_score(model, X_spline, y, cv=5, scoring='r2')
                
                model.fit(X_spline, y)
                y_pred = model.predict(X_spline)
                
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                mae = mean_absolute_error(y, y_pred)
                
                print(f"R²: {r2:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"交叉验证R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
                results[n_knots] = {
                    'model': model,
                    'spline_transformer': spline_transformer,
                    'scaler': scaler,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'y_pred': y_pred
                }
                
                # 更新最佳模型
                if cv_scores.mean() > self.best_score:
                    self.best_score = cv_scores.mean()
                    self.best_model = f"样条回归({n_knots}节点)"
                
            except Exception as e:
                print(f"样条回归失败 (节点数: {n_knots}): {e}")
                continue
        
        self.results['spline'] = results
        return results
    
    def machine_learning_analysis(self):
        """机器学习模型分析"""
        print(f"\n=== 机器学习模型分析 ===")
        
        X = self.df[['检测孕周', '孕妇BMI', '年龄']].values
        y = self.df['Y染色体浓度'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        models = {
            'Ridge回归': Ridge(alpha=1.0),
            'Lasso回归': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
            '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42),
            '支持向量机': SVR(kernel='rbf', C=1.0, gamma='scale'),
            '决策树': DecisionTreeRegressor(random_state=42),
            '神经网络': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n--- {name} ---")
            
            try:
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
                
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                mae = mean_absolute_error(y, y_pred)
                
                print(f"R²: {r2:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"交叉验证R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
                results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'y_pred': y_pred
                }
                
                # 更新最佳模型
                if cv_scores.mean() > self.best_score:
                    self.best_score = cv_scores.mean()
                    self.best_model = name
                
            except Exception as e:
                print(f"{name}训练失败: {e}")
                continue
        
        self.results['ml_models'] = results
        return results
    
    def ensemble_learning_analysis(self):
        """集成学习分析"""
        print(f"\n=== 集成学习分析 ===")
        
        X = self.df_eng[self.feature_cols].values
        y = self.df_eng['Y染色体浓度'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        # 1. 投票回归器
        print("\n--- 投票回归器 ---")
        
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
        voting_soft = VotingRegressor(models, weights=[1, 1, 1, 2, 2])
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
                
                # 更新最佳模型
                if cv_scores.mean() > self.best_score:
                    self.best_score = cv_scores.mean()
                    self.best_model = name
                    
            except Exception as e:
                print(f"{name}训练失败: {e}")
        
        self.results['ensemble'] = results
        return results
    
    def clustering_analysis(self):
        """聚类分析"""
        print("\n=== 聚类分析 ===")
        
        X = self.df_eng[self.feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        # K-means聚类
        print("\n--- K-means聚类 ---")
        kmeans_results = {}
        
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
            
            print(f"  {n_clusters}个聚类: 轮廓系数={silhouette_avg:.3f}, CH指数={calinski_harabasz:.3f}")
            
            kmeans_results[n_clusters] = {
                'model': kmeans,
                'labels': cluster_labels,
                'silhouette_score': silhouette_avg,
                'calinski_harabasz_score': calinski_harabasz
            }
        
        best_k = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette_score'])
        print(f"最佳K值: {best_k} (轮廓系数: {kmeans_results[best_k]['silhouette_score']:.3f})")
        
        results['kmeans'] = kmeans_results
        results['best_k'] = best_k
        
        self.results['clustering'] = results
        return results
    
    def dimensionality_reduction_analysis(self):
        """降维分析"""
        print("\n=== 降维分析 ===")
        
        X = self.df_eng[self.feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        # PCA分析
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
        
        # t-SNE分析
        print("\n--- t-SNE分析 ---")
        tsne_results = {}
        
        perplexities = [5, 10, 20, 30, 50]
        
        for perplexity in perplexities:
            if perplexity < X_scaled.shape[0] / 3:
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                X_tsne = tsne.fit_transform(X_scaled)
                
                print(f"  perplexity={perplexity}: 降维到2D完成")
                
                tsne_results[perplexity] = {
                    'model': tsne,
                    'transformed_data': X_tsne
                }
        
        results['tsne'] = tsne_results
        
        self.results['dim_reduction'] = results
        return results
    
    def create_comprehensive_visualizations(self):
        """创建综合可视化图表"""
        print("\n=== 创建综合可视化图表 ===")
        
        # 设置中文字体
        font_priority = self.setup_chinese_font()
        
        # 设置图表样式
        plt.style.use('default')
        
        # 创建大图
        fig = plt.figure(figsize=(24, 20))
        
        # 1. 原始数据散点图
        plt.subplot(4, 4, 1)
        plt.scatter(self.df['检测孕周'], self.df['Y染色体浓度'], alpha=0.6, s=20, color='blue')
        plt.xlabel('检测孕周 (周)', fontsize=12, fontweight='bold')
        plt.ylabel('Y染色体浓度 (%)', fontsize=12, fontweight='bold')
        plt.title('Y染色体浓度与孕周关系', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        
        # 2. BMI散点图
        plt.subplot(4, 4, 2)
        plt.scatter(self.df['孕妇BMI'], self.df['Y染色体浓度'], alpha=0.6, s=20, color='orange')
        plt.xlabel('孕妇BMI (kg/m²)', fontsize=12, fontweight='bold')
        plt.ylabel('Y染色体浓度 (%)', fontsize=12, fontweight='bold')
        plt.title('Y染色体浓度与BMI关系', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        
        # 3. 年龄散点图
        plt.subplot(4, 4, 3)
        plt.scatter(self.df['年龄'], self.df['Y染色体浓度'], alpha=0.6, s=20, color='green')
        plt.xlabel('年龄 (岁)', fontsize=12, fontweight='bold')
        plt.ylabel('Y染色体浓度 (%)', fontsize=12, fontweight='bold')
        plt.title('Y染色体浓度与年龄关系', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        
        # 4. 模型性能对比
        plt.subplot(4, 4, 4)
        if 'ml_models' in self.results:
            model_names = []
            r2_scores = []
            
            for name, result in self.results['ml_models'].items():
                model_names.append(name)
                r2_scores.append(result['cv_r2_mean'])
            
            bars = plt.bar(range(len(model_names)), r2_scores, 
                          color=['lightblue', 'lightgreen', 'orange', 'lightcoral', 'pink', 'lightyellow', 'lightcyan', 'lightsteelblue'])
            plt.xlabel('模型类型', fontsize=12, fontweight='bold')
            plt.ylabel('交叉验证R²', fontsize=12, fontweight='bold')
            plt.title('模型性能对比', fontsize=14, fontweight='bold', pad=20)
            plt.xticks(range(len(model_names)), model_names, rotation=45, fontsize=8)
            plt.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, (bar, score) in enumerate(zip(bars, r2_scores)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 5-8. 最佳模型预测结果
        if self.best_model and 'ml_models' in self.results:
            best_result = None
            for name, result in self.results['ml_models'].items():
                if name == self.best_model:
                    best_result = result
                    break
            
            if best_result:
                y_pred = best_result['y_pred']
                
                # 预测 vs 实际
                plt.subplot(4, 4, 5)
                plt.scatter(self.df['Y染色体浓度'], y_pred, alpha=0.6, s=20)
                plt.plot([self.df['Y染色体浓度'].min(), self.df['Y染色体浓度'].max()], 
                        [self.df['Y染色体浓度'].min(), self.df['Y染色体浓度'].max()], 'r--', linewidth=2)
                plt.xlabel('实际Y染色体浓度 (%)', fontsize=12, fontweight='bold')
                plt.ylabel('预测Y染色体浓度 (%)', fontsize=12, fontweight='bold')
                plt.title(f'{self.best_model} - 预测vs实际', fontsize=14, fontweight='bold', pad=20)
                plt.grid(True, alpha=0.3)
                
                # 残差图
                plt.subplot(4, 4, 6)
                residuals = self.df['Y染色体浓度'] - y_pred
                plt.scatter(y_pred, residuals, alpha=0.6, s=20)
                plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
                plt.xlabel('预测值', fontsize=12, fontweight='bold')
                plt.ylabel('残差', fontsize=12, fontweight='bold')
                plt.title(f'{self.best_model} - 残差图', fontsize=14, fontweight='bold', pad=20)
                plt.grid(True, alpha=0.3)
                
                # 残差分布
                plt.subplot(4, 4, 7)
                plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.xlabel('残差', fontsize=12, fontweight='bold')
                plt.ylabel('频数', fontsize=12, fontweight='bold')
                plt.title(f'{self.best_model} - 残差分布', fontsize=14, fontweight='bold', pad=20)
                plt.grid(True, alpha=0.3)
                
                # Q-Q图
                plt.subplot(4, 4, 8)
                from scipy import stats
                stats.probplot(residuals, dist="norm", plot=plt)
                plt.title(f'{self.best_model} - Q-Q图', fontsize=14, fontweight='bold', pad=20)
                plt.grid(True, alpha=0.3)
        
        # 9-12. 聚类结果
        if 'clustering' in self.results and 'kmeans' in self.results['clustering']:
            best_k = self.results['clustering']['best_k']
            kmeans_result = self.results['clustering']['kmeans'][best_k]
            
            # K-means聚类结果
            plt.subplot(4, 4, 9)
            if 'dim_reduction' in self.results and 'pca' in self.results['dim_reduction'] and 2 in self.results['dim_reduction']['pca']:
                pca_data = self.results['dim_reduction']['pca'][2]['transformed_data']
                scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_result['labels'], 
                                    cmap='viridis', alpha=0.6, s=20)
                plt.colorbar(scatter, label='聚类标签')
                plt.xlabel('第一主成分', fontsize=12, fontweight='bold')
                plt.ylabel('第二主成分', fontsize=12, fontweight='bold')
                plt.title(f'K-means聚类结果 (K={best_k})', fontsize=14, fontweight='bold', pad=20)
            else:
                plt.scatter(self.df['检测孕周'], self.df['Y染色体浓度'], c=kmeans_result['labels'], 
                           cmap='viridis', alpha=0.6, s=20)
                plt.xlabel('检测孕周 (周)', fontsize=12, fontweight='bold')
                plt.ylabel('Y染色体浓度 (%)', fontsize=12, fontweight='bold')
                plt.title(f'K-means聚类结果 (K={best_k})', fontsize=14, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3)
            
            # 聚类轮廓系数
            plt.subplot(4, 4, 10)
            k_values = list(self.results['clustering']['kmeans'].keys())
            silhouette_scores = [self.results['clustering']['kmeans'][k]['silhouette_score'] for k in k_values]
            
            plt.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('聚类数 K', fontsize=12, fontweight='bold')
            plt.ylabel('轮廓系数', fontsize=12, fontweight='bold')
            plt.title('K-means聚类轮廓系数', fontsize=14, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3)
            
            # 标记最佳K值
            plt.plot(best_k, self.results['clustering']['kmeans'][best_k]['silhouette_score'], 'ro', markersize=12, label=f'最佳K={best_k}')
            plt.legend()
        
        # 13-16. PCA降维结果
        if 'dim_reduction' in self.results and 'pca' in self.results['dim_reduction']:
            # PCA降维结果
            plt.subplot(4, 4, 13)
            if 2 in self.results['dim_reduction']['pca']:
                pca_data = self.results['dim_reduction']['pca'][2]['transformed_data']
                plt.scatter(pca_data[:, 0], pca_data[:, 1], c=self.df['Y染色体浓度'], 
                           cmap='viridis', alpha=0.6, s=20)
                plt.colorbar(label='Y染色体浓度 (%)')
                plt.xlabel('第一主成分', fontsize=12, fontweight='bold')
                plt.ylabel('第二主成分', fontsize=12, fontweight='bold')
                plt.title('PCA降维结果', fontsize=14, fontweight='bold', pad=20)
                plt.grid(True, alpha=0.3)
            
            # PCA解释方差比
            plt.subplot(4, 4, 14)
            pca_results = self.results['dim_reduction']['pca']
            n_components = list(pca_results.keys())
            explained_variance = [pca_results[n]['cumulative_variance'][-1] for n in n_components]
            
            plt.plot(n_components, explained_variance, 'go-', linewidth=2, markersize=8)
            plt.xlabel('主成分数', fontsize=12, fontweight='bold')
            plt.ylabel('累积解释方差比', fontsize=12, fontweight='bold')
            plt.title('PCA解释方差比', fontsize=14, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3)
        
        # 调整布局和保存
        plt.tight_layout(pad=3.0)
        
        # 保存图表前再次确保字体设置
        try:
            # 重新设置字体确保保存时正确
            plt.rcParams['font.sans-serif'] = font_priority
            plt.rcParams['axes.unicode_minus'] = False
            
            plt.savefig('unified_algorithm_evaluation_v1.3.png', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print("统一算法评估可视化图表已保存为: unified_algorithm_evaluation_v1.3.png")
        except Exception as e:
            print(f"保存图表时出错: {e}")
            try:
                plt.savefig('unified_algorithm_evaluation_v1.3.png', dpi=150, bbox_inches='tight')
                print("使用较低分辨率保存成功")
            except Exception as e2:
                print(f"保存图表完全失败: {e2}")
        
        # 显示图表
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错: {e}")
            print("图表已保存，但无法在终端中显示")
        
        return fig
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n=== 生成综合报告 ===")
        
        report = []
        report.append("# 统一算法评估报告 v1.3")
        report.append("=" * 60)
        report.append("")
        
        # 数据概况
        report.append("## 数据概况")
        report.append(f"- 样本数: {len(self.df)}")
        report.append(f"- 特征数: {self.df.shape[1]}")
        report.append(f"- 目标变量: Y染色体浓度")
        report.append("")
        
        # 最佳模型
        report.append("## 最佳模型")
        report.append(f"- 模型类型: {self.best_model}")
        report.append(f"- 交叉验证R²: {self.best_score:.4f}")
        report.append("")
        
        # 模型性能对比
        if 'ml_models' in self.results:
            report.append("## 机器学习模型性能对比")
            report.append("")
            
            for name, result in self.results['ml_models'].items():
                report.append(f"- {name}: R²={result['r2']:.4f}, 交叉验证R²={result['cv_r2_mean']:.4f}±{result['cv_r2_std']:.4f}")
            report.append("")
        
        # 多项式回归结果
        if 'polynomial' in self.results:
            report.append("## 多项式回归结果")
            report.append("")
            
            for degree, result in self.results['polynomial'].items():
                report.append(f"- {degree}次多项式: R²={result['r2']:.4f}, 交叉验证R²={result['cv_r2_mean']:.4f}±{result['cv_r2_std']:.4f}")
            report.append("")
        
        # 样条回归结果
        if 'spline' in self.results:
            report.append("## 样条回归结果")
            report.append("")
            
            for n_knots, result in self.results['spline'].items():
                report.append(f"- {n_knots}节点样条: R²={result['r2']:.4f}, 交叉验证R²={result['cv_r2_mean']:.4f}±{result['cv_r2_std']:.4f}")
            report.append("")
        
        # 集成学习结果
        if 'ensemble' in self.results:
            report.append("## 集成学习结果")
            report.append("")
            
            if 'voting' in self.results['ensemble']:
                hard_result = self.results['ensemble']['voting']['hard']
                soft_result = self.results['ensemble']['voting']['soft']
                
                report.append(f"- 硬投票: R²={hard_result['r2']:.4f}, RMSE={hard_result['rmse']:.4f}, 交叉验证R²={hard_result['cv_r2']:.4f}")
                report.append(f"- 软投票: R²={soft_result['r2']:.4f}, RMSE={soft_result['rmse']:.4f}, 交叉验证R²={soft_result['cv_r2']:.4f}")
                report.append("")
            
            for name, result in self.results['ensemble'].items():
                if name != 'voting':
                    report.append(f"- {name}: R²={result['r2']:.4f}, RMSE={result['rmse']:.4f}, 交叉验证R²={result['cv_r2']:.4f}")
            report.append("")
        
        # 聚类分析结果
        if 'clustering' in self.results and 'kmeans' in self.results['clustering']:
            report.append("## 聚类分析结果")
            report.append("")
            
            best_k = self.results['clustering']['best_k']
            kmeans_result = self.results['clustering']['kmeans'][best_k]
            
            report.append(f"- 最佳聚类数: {best_k}")
            report.append(f"- 轮廓系数: {kmeans_result['silhouette_score']:.3f}")
            report.append(f"- CH指数: {kmeans_result['calinski_harabasz_score']:.3f}")
            report.append("")
        
        # 降维分析结果
        if 'dim_reduction' in self.results and 'pca' in self.results['dim_reduction']:
            report.append("## 降维分析结果")
            report.append("")
            
            report.append("### PCA分析")
            for n_components, result in self.results['dim_reduction']['pca'].items():
                cumulative_var = result['cumulative_variance'][-1]
                report.append(f"- {n_components}个主成分: 累积解释方差比={cumulative_var:.3f}")
            report.append("")
        
        # 建议
        report.append("## 建议")
        report.append("1. 根据交叉验证结果选择最佳模型")
        report.append("2. 考虑特征工程和特征选择")
        report.append("3. 使用集成方法提高预测精度")
        report.append("4. 结合聚类分析发现数据模式")
        report.append("5. 使用降维方法减少特征维度")
        report.append("")
        
        # 保存报告
        report_text = "\n".join(report)
        try:
            with open('unified_algorithm_evaluation_report_v1.3.md', 'w', encoding='utf-8') as f:
                f.write(report_text)
            print("统一算法评估报告已保存为: unified_algorithm_evaluation_report_v1.3.md")
        except Exception as e:
            print(f"保存报告时出错: {e}")
            # 尝试使用GBK编码
            try:
                with open('unified_algorithm_evaluation_report_v1.3.md', 'w', encoding='gbk') as f:
                    f.write(report_text)
                print("使用GBK编码保存报告成功")
            except Exception as e2:
                print(f"使用GBK编码保存报告也失败: {e2}")
        
        print("\n报告内容:")
        print(report_text)
        
        return report_text
    
    def run_complete_evaluation(self):
        """运行完整的算法评估"""
        print("统一算法评估系统 v1.3")
        print("整合所有机器学习算法，提供全面的模型评估和拟合结果显示")
        print("=" * 80)
        
        # 1. 加载数据
        if not self.load_data():
            return False
        
        # 2. 数据探索
        self.explore_data()
        
        # 3. 特征工程
        self.feature_engineering()
        
        # 4. 相关性分析
        self.correlation_analysis()
        
        # 5. 多项式回归分析
        self.polynomial_regression_analysis()
        
        # 6. 样条回归分析
        self.spline_regression_analysis()
        
        # 7. 机器学习模型分析
        self.machine_learning_analysis()
        
        # 8. 集成学习分析
        self.ensemble_learning_analysis()
        
        # 9. 聚类分析
        self.clustering_analysis()
        
        # 10. 降维分析
        self.dimensionality_reduction_analysis()
        
        # 11. 创建综合可视化
        self.create_comprehensive_visualizations()
        
        # 12. 生成综合报告
        self.generate_comprehensive_report()
        
        print(f"\n=== 统一算法评估完成 ===")
        print(f"最佳模型: {self.best_model}")
        print(f"最佳交叉验证R²: {self.best_score:.4f}")
        print("所有结果已保存到当前目录")
        
        return True

def main():
    """主函数"""
    evaluator = UnifiedAlgorithmEvaluator()
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()
