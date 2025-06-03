import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns
from config.settings import Config

class RecommendationVisualizer:
    """推荐系统可视化工具"""
    
    def plot_data_analysis(self, df):
        """绘制数据分析可视化"""
        plt.figure(figsize=Config.FIGURE_SIZE, dpi=Config.DPI)
        
        plt.subplot(1, 3, 1)
        df['rating'].value_counts().sort_index().plot(kind='bar')
        plt.title('评分分布')
        plt.xlabel('评分')
        plt.ylabel('频次')
        
        plt.subplot(1, 3, 2)
        user_ratings = df.groupby('user').size()
        plt.hist(user_ratings, bins=50, alpha=0.7)
        plt.title('用户评分数量分布')
        plt.xlabel('用户评分数量')
        plt.ylabel('用户数量')
        
        plt.subplot(1, 3, 3)
        item_ratings = df.groupby('item').size()
        plt.hist(item_ratings, bins=50, alpha=0.7)
        plt.title('物品评分数量分布')
        plt.xlabel('物品评分数量')
        plt.ylabel('物品数量')
        
        plt.tight_layout()
        plt.show()
        
    def plot_algorithm_comparison(self, results):
        """绘制算法性能比较"""
        plt.figure(figsize=Config.FIGURE_SIZE, dpi=Config.DPI)
        
        algorithms = list(results.keys())
        rmse_scores = [results[alg]['test_rmse'] for alg in algorithms]
        mae_scores = [results[alg]['test_mae'] for alg in algorithms]
        
        # RMSE比较
        plt.subplot(1, 2, 1)
        plt.bar(algorithms, rmse_scores, color='skyblue', alpha=0.7)
        plt.title('算法 RMSE 比较')
        plt.ylabel('RMSE')
        plt.tick_params(axis='x', rotation=45)
        
        # MAE比较
        plt.subplot(1, 2, 2)
        plt.bar(algorithms, mae_scores, color='lightcoral', alpha=0.7)
        plt.title('算法 MAE 比较')
        plt.ylabel('MAE')
        plt.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()