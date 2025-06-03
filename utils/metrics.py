import numpy as np
import pandas as pd
from surprise import accuracy

class RecommendationMetrics:
    """推荐系统评估指标"""
    
    def calculate_dataset_stats(self, df):
        """计算数据集统计指标"""
        n_users = df['user'].nunique()
        n_items = df['item'].nunique()
        n_ratings = len(df)
        sparsity = (1 - n_ratings / (n_users * n_items)) * 100
        
        return {
            "数据集大小": f"{n_ratings:,} 条评分记录",
            "用户数量": f"{n_users:,}",
            "物品数量": f"{n_items:,}",
            "评分范围": f"{df['rating'].min()} - {df['rating'].max()}",
            "平均评分": f"{df['rating'].mean():.2f}",
            "数据稀疏度": f"{sparsity:.2f}%"
        }
        
    def evaluate_algorithm(self, testset, predictions):
        """评估算法性能"""
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        return {
            'test_rmse': rmse,
            'test_mae': mae
        }