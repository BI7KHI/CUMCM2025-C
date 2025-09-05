#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1 分析代码 v1.3 - 高级拟合算法版本
基于清洗后的男性胎儿数据，使用多种高级拟合算法提高拟合程度
包括：多项式回归、样条回归、分段回归、随机森林、支持向量机等
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
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
        # 读取清洗后的CSV文件
        df = pd.read_csv('../Source_DATA/male_fetal_data_cleaned.csv')
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

def advanced_correlation_analysis(df):
    """高级相关性分析"""
    print("\n=== 高级相关性分析 ===")
    
    # 选择关键变量
    key_vars = ['检测孕周', '孕妇BMI', 'Y染色体浓度', '年龄']
    corr_data = df[key_vars].dropna()
    
    print(f"相关性分析样本数: {len(corr_data)}")
    
    # 计算多种相关系数
    correlations = {}
    
    # 1. Pearson相关系数
    print("\n1. Pearson相关系数 (线性关系):")
    pearson_corr = corr_data.corr(method='pearson')
    print(pearson_corr.round(4))
    
    # 2. Spearman相关系数
    print("\n2. Spearman相关系数 (单调关系):")
    spearman_corr = corr_data.corr(method='spearman')
    print(spearman_corr.round(4))
    
    # 3. 与Y染色体浓度的详细相关性
    print("\n3. 与Y染色体浓度的详细相关性:")
    y_col = 'Y染色体浓度'
    
    for var in ['检测孕周', '孕妇BMI', '年龄']:
        if var in corr_data.columns:
            # Pearson相关
            pearson_r, pearson_p = pearsonr(corr_data[var], corr_data[y_col])
            # Spearman相关
            spearman_r, spearman_p = spearmanr(corr_data[var], corr_data[y_col])
            
            print(f"\n{var} vs {y_col}:")
            print(f"  Pearson: r={pearson_r:.4f}, p={pearson_p:.6f}")
            print(f"  Spearman: ρ={spearman_r:.4f}, p={spearman_p:.6f}")
            
            # 显著性判断
            if pearson_p < 0.001:
                pearson_sig = "极显著 (p < 0.001)"
            elif pearson_p < 0.01:
                pearson_sig = "高度显著 (p < 0.01)"
            elif pearson_p < 0.05:
                pearson_sig = "显著 (p < 0.05)"
            else:
                pearson_sig = "不显著 (p ≥ 0.05)"
            
            if spearman_p < 0.001:
                spearman_sig = "极显著 (p < 0.001)"
            elif spearman_p < 0.01:
                spearman_sig = "高度显著 (p < 0.01)"
            elif spearman_p < 0.05:
                spearman_sig = "显著 (p < 0.05)"
            else:
                spearman_sig = "不显著 (p ≥ 0.05)"
            
            print(f"  Pearson显著性: {pearson_sig}")
            print(f"  Spearman显著性: {spearman_sig}")
    
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'data': corr_data
    }

def polynomial_regression_analysis(df, max_degree=5):
    """多项式回归分析"""
    print(f"\n=== 多项式回归分析 (最高{max_degree}次) ===")
    
    # 准备数据
    X = df[['检测孕周', '孕妇BMI', '年龄']].values
    y = df['Y染色体浓度'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for degree in range(1, max_degree + 1):
        print(f"\n--- {degree}次多项式回归 ---")
        
        # 创建多项式特征
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(X_scaled)
        
        # 交叉验证
        model = LinearRegression()
        cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
        
        # 训练模型
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        # 评估指标
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
    
    # 选择最佳模型
    best_degree = max(results.keys(), key=lambda k: results[k]['cv_r2_mean'])
    print(f"\n最佳多项式次数: {best_degree} (交叉验证R²: {results[best_degree]['cv_r2_mean']:.4f})")
    
    return results, best_degree

def spline_regression_analysis(df, n_knots_list=[3, 5, 7, 10]):
    """样条回归分析"""
    print(f"\n=== 样条回归分析 ===")
    
    # 准备数据
    X = df[['检测孕周', '孕妇BMI', '年龄']].values
    y = df['Y染色体浓度'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for n_knots in n_knots_list:
        print(f"\n--- 样条回归 (节点数: {n_knots}) ---")
        
        try:
            # 创建样条特征
            spline_transformer = SplineTransformer(n_knots=n_knots, degree=3)
            X_spline = spline_transformer.fit_transform(X_scaled)
            
            # 交叉验证
            model = LinearRegression()
            cv_scores = cross_val_score(model, X_spline, y, cv=5, scoring='r2')
            
            # 训练模型
            model.fit(X_spline, y)
            y_pred = model.predict(X_spline)
            
            # 评估指标
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
            
        except Exception as e:
            print(f"样条回归失败 (节点数: {n_knots}): {e}")
            continue
    
    if results:
        # 选择最佳模型
        best_n_knots = max(results.keys(), key=lambda k: results[k]['cv_r2_mean'])
        print(f"\n最佳样条节点数: {best_n_knots} (交叉验证R²: {results[best_n_knots]['cv_r2_mean']:.4f})")
        return results, best_n_knots
    else:
        print("所有样条回归都失败了")
        return {}, None

def machine_learning_models_analysis(df):
    """机器学习模型分析"""
    print(f"\n=== 机器学习模型分析 ===")
    
    # 准备数据
    X = df[['检测孕周', '孕妇BMI', '年龄']].values
    y = df['Y染色体浓度'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 定义模型
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
            # 交叉验证
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            
            # 训练模型
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            
            # 评估指标
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
            
        except Exception as e:
            print(f"{name}训练失败: {e}")
            continue
    
    if results:
        # 选择最佳模型
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_r2_mean'])
        print(f"\n最佳机器学习模型: {best_model_name} (交叉验证R²: {results[best_model_name]['cv_r2_mean']:.4f})")
        return results, best_model_name
    else:
        print("所有机器学习模型都失败了")
        return {}, None

def hyperparameter_optimization(df, model_name='RandomForest'):
    """超参数优化"""
    print(f"\n=== {model_name} 超参数优化 ===")
    
    # 准备数据
    X = df[['检测孕周', '孕妇BMI', '年龄']].values
    y = df['Y染色体浓度'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 定义参数网格
    if model_name == 'RandomForest':
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=42)
    elif model_name == 'GradientBoosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        model = GradientBoostingRegressor(random_state=42)
    elif model_name == 'SVR':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        }
        model = SVR()
    else:
        print(f"不支持的模型: {model_name}")
        return None
    
    # 网格搜索
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='r2', 
        n_jobs=-1, verbose=1
    )
    
    print("开始网格搜索...")
    grid_search.fit(X_scaled, y)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证R²: {grid_search.best_score_:.4f}")
    
    # 使用最佳参数训练模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_scaled)
    
    # 最终评估
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    print(f"最终R²: {r2:.4f}")
    print(f"最终RMSE: {rmse:.4f}")
    print(f"最终MAE: {mae:.4f}")
    
    return {
        'best_model': best_model,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'y_pred': y_pred,
        'scaler': scaler
    }

def create_advanced_visualizations(df, model_results):
    """创建高级可视化图表"""
    print("\n=== 创建高级可视化图表 ===")
    
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
    fig = plt.figure(figsize=(24, 18))
    
    # 1. 原始数据散点图
    plt.subplot(3, 4, 1)
    plt.scatter(df['检测孕周'], df['Y染色体浓度'], alpha=0.6, s=20, color='blue')
    plt.xlabel('检测孕周 (周)', fontsize=12, fontweight='bold')
    plt.ylabel('Y染色体浓度 (%)', fontsize=12, fontweight='bold')
    plt.title('Y染色体浓度与孕周关系', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # 2. BMI散点图
    plt.subplot(3, 4, 2)
    plt.scatter(df['孕妇BMI'], df['Y染色体浓度'], alpha=0.6, s=20, color='orange')
    plt.xlabel('孕妇BMI (kg/m²)', fontsize=12, fontweight='bold')
    plt.ylabel('Y染色体浓度 (%)', fontsize=12, fontweight='bold')
    plt.title('Y染色体浓度与BMI关系', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # 3. 年龄散点图
    plt.subplot(3, 4, 3)
    plt.scatter(df['年龄'], df['Y染色体浓度'], alpha=0.6, s=20, color='green')
    plt.xlabel('年龄 (岁)', fontsize=12, fontweight='bold')
    plt.ylabel('Y染色体浓度 (%)', fontsize=12, fontweight='bold')
    plt.title('Y染色体浓度与年龄关系', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # 4. 模型性能对比
    plt.subplot(3, 4, 4)
    if 'polynomial' in model_results and 'ml_models' in model_results:
        model_names = []
        r2_scores = []
        
        # 多项式模型
        for degree, result in model_results['polynomial'].items():
            model_names.append(f'多项式{degree}次')
            r2_scores.append(result['cv_r2_mean'])
        
        # 机器学习模型
        for name, result in model_results['ml_models'].items():
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
    if 'best_model' in model_results:
        best_model_name = model_results['best_model_name']
        best_result = model_results['best_result']
        y_pred = best_result['y_pred']
        
        # 预测 vs 实际
        plt.subplot(3, 4, 5)
        plt.scatter(df['Y染色体浓度'], y_pred, alpha=0.6, s=20)
        plt.plot([df['Y染色体浓度'].min(), df['Y染色体浓度'].max()], 
                [df['Y染色体浓度'].min(), df['Y染色体浓度'].max()], 'r--', linewidth=2)
        plt.xlabel('实际Y染色体浓度 (%)', fontsize=12, fontweight='bold')
        plt.ylabel('预测Y染色体浓度 (%)', fontsize=12, fontweight='bold')
        plt.title(f'{best_model_name} - 预测vs实际', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        
        # 残差图
        plt.subplot(3, 4, 6)
        residuals = df['Y染色体浓度'] - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('预测值', fontsize=12, fontweight='bold')
        plt.ylabel('残差', fontsize=12, fontweight='bold')
        plt.title(f'{best_model_name} - 残差图', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        
        # 残差分布
        plt.subplot(3, 4, 7)
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('残差', fontsize=12, fontweight='bold')
        plt.ylabel('频数', fontsize=12, fontweight='bold')
        plt.title(f'{best_model_name} - 残差分布', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        
        # Q-Q图
        plt.subplot(3, 4, 8)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'{best_model_name} - Q-Q图', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
    
    # 9-12. 特征重要性（如果可用）
    if 'best_model' in model_results and hasattr(model_results['best_result']['model'], 'feature_importances_'):
        plt.subplot(3, 4, 9)
        feature_names = ['检测孕周', '孕妇BMI', '年龄']
        importances = model_results['best_result']['model'].feature_importances_
        
        bars = plt.bar(feature_names, importances, color=['lightblue', 'lightgreen', 'orange'])
        plt.xlabel('特征', fontsize=12, fontweight='bold')
        plt.ylabel('重要性', fontsize=12, fontweight='bold')
        plt.title(f'{model_results["best_model_name"]} - 特征重要性', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, importance in zip(bars, importances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{importance:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 调整布局和保存
    plt.tight_layout(pad=3.0)
    
    # 保存图表
    try:
        plt.savefig('t1_analysis_advanced_v1.3.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print("高级可视化图表已保存为: t1_analysis_advanced_v1.3.png")
    except Exception as e:
        print(f"保存图表时出错: {e}")
        plt.savefig('t1_analysis_advanced_v1.3.png', dpi=150, bbox_inches='tight')
    
    # 显示图表
    try:
        plt.show()
    except Exception as e:
        print(f"显示图表时出错: {e}")
        print("图表已保存，但无法在终端中显示")
    
    return fig

def generate_comprehensive_report(model_results, df):
    """生成综合报告"""
    print("\n=== 生成综合报告 ===")
    
    report = []
    report.append("# T1分析高级拟合算法报告 v1.3")
    report.append("=" * 50)
    report.append("")
    
    # 数据概况
    report.append("## 数据概况")
    report.append(f"- 样本数: {len(df)}")
    report.append(f"- 特征数: 3 (检测孕周, 孕妇BMI, 年龄)")
    report.append(f"- 目标变量: Y染色体浓度")
    report.append("")
    
    # 模型性能对比
    if 'polynomial' in model_results and 'ml_models' in model_results:
        report.append("## 模型性能对比")
        report.append("")
        
        # 多项式模型
        report.append("### 多项式回归模型")
        for degree, result in model_results['polynomial'].items():
            report.append(f"- {degree}次多项式: R²={result['r2']:.4f}, 交叉验证R²={result['cv_r2_mean']:.4f}±{result['cv_r2_std']:.4f}")
        report.append("")
        
        # 机器学习模型
        report.append("### 机器学习模型")
        for name, result in model_results['ml_models'].items():
            report.append(f"- {name}: R²={result['r2']:.4f}, 交叉验证R²={result['cv_r2_mean']:.4f}±{result['cv_r2_std']:.4f}")
        report.append("")
    
    # 最佳模型
    if 'best_model' in model_results:
        report.append("## 最佳模型")
        report.append(f"- 模型类型: {model_results['best_model_name']}")
        report.append(f"- R²: {model_results['best_result']['r2']:.4f}")
        report.append(f"- RMSE: {model_results['best_result']['rmse']:.4f}")
        report.append(f"- MAE: {model_results['best_result']['mae']:.4f}")
        report.append("")
    
    # 超参数优化结果
    if 'hyperopt' in model_results:
        report.append("## 超参数优化结果")
        report.append(f"- 最佳参数: {model_results['hyperopt']['best_params']}")
        report.append(f"- 最佳交叉验证R²: {model_results['hyperopt']['best_score']:.4f}")
        report.append("")
    
    # 建议
    report.append("## 建议")
    report.append("1. 根据交叉验证结果选择最佳模型")
    report.append("2. 考虑特征工程和特征选择")
    report.append("3. 使用集成方法提高预测精度")
    report.append("4. 定期重新训练模型以适应新数据")
    report.append("")
    
    # 保存报告
    report_text = "\n".join(report)
    with open('t1_analysis_report_v1.3.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("综合报告已保存为: t1_analysis_report_v1.3.md")
    print("\n报告内容:")
    print(report_text)
    
    return report_text

def main():
    """主函数"""
    print("T1 分析代码 v1.3 - 高级拟合算法版本")
    print("基于清洗后的男性胎儿数据，使用多种高级拟合算法提高拟合程度")
    print("=" * 80)
    
    # 1. 加载数据
    df = load_cleaned_data()
    if df is None:
        return
    
    # 2. 数据探索
    df = explore_cleaned_data(df)
    
    # 3. 高级相关性分析
    corr_results = advanced_correlation_analysis(df)
    
    # 4. 多项式回归分析
    poly_results, best_poly_degree = polynomial_regression_analysis(df, max_degree=5)
    
    # 5. 样条回归分析
    spline_results, best_spline_knots = spline_regression_analysis(df)
    
    # 6. 机器学习模型分析
    ml_results, best_ml_model = machine_learning_models_analysis(df)
    
    # 7. 超参数优化（选择最佳机器学习模型）
    hyperopt_results = None
    if best_ml_model:
        hyperopt_results = hyperparameter_optimization(df, best_ml_model)
    
    # 8. 整合结果
    model_results = {
        'polynomial': poly_results,
        'spline': spline_results,
        'ml_models': ml_results,
        'hyperopt': hyperopt_results
    }
    
    # 确定最佳模型
    best_overall_model = None
    best_overall_score = -np.inf
    
    # 比较所有模型
    for degree, result in poly_results.items():
        if result['cv_r2_mean'] > best_overall_score:
            best_overall_score = result['cv_r2_mean']
            best_overall_model = f"多项式{degree}次"
            model_results['best_model'] = best_overall_model
            model_results['best_result'] = result
    
    for name, result in ml_results.items():
        if result['cv_r2_mean'] > best_overall_score:
            best_overall_score = result['cv_r2_mean']
            best_overall_model = name
            model_results['best_model'] = best_overall_model
            model_results['best_result'] = result
    
    if hyperopt_results and hyperopt_results['best_score'] > best_overall_score:
        best_overall_score = hyperopt_results['best_score']
        best_overall_model = f"{best_ml_model}(优化)"
        model_results['best_model'] = best_overall_model
        model_results['best_result'] = hyperopt_results
    
    model_results['best_model_name'] = best_overall_model
    print(f"\n最佳模型: {best_overall_model} (交叉验证R²: {best_overall_score:.4f})")
    
    # 9. 创建高级可视化
    create_advanced_visualizations(df, model_results)
    
    # 10. 生成综合报告
    generate_comprehensive_report(model_results, df)
    
    print("\n=== 分析完成 ===")
    print("所有结果已保存到当前目录")

if __name__ == "__main__":
    main()
