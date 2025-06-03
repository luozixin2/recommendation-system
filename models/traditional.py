from surprise import SVD, KNNBasic, KNNWithMeans, NMF, BaselineOnly
from surprise.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)

class TraditionalRecommender:
    """传统推荐算法管理类"""
    
    def __init__(self):
        self.algorithms = {}
        self.setup_algorithms()
        
    def setup_algorithms(self):
        """设置传统推荐算法"""
        self.algorithms = {
            'SVD': SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, verbose=False),
            '基于用户的协同过滤': KNNWithMeans(
                k=40, sim_options={'name': 'cosine', 'user_based': True}, verbose=False
            ),
            '基于物品的协同过滤': KNNWithMeans(
                k=40, sim_options={'name': 'cosine', 'user_based': False}, verbose=False
            ),
            'NMF': NMF(n_factors=50, n_epochs=50, verbose=False),
            'Baseline': BaselineOnly(verbose=False)
        }
        
        logger.info(f"传统算法设置完成，共 {len(self.algorithms)} 个算法")
        
    def get_algorithms(self):
        """获取所有算法"""
        return self.algorithms
    
    def hyperparameter_tuning(self, data, algorithm_name='SVD'):
        """超参数调优"""
        logger.info(f"开始对 {algorithm_name} 进行超参数调优")
        
        if algorithm_name == 'SVD':
            param_grid = {
                'n_factors': [50, 100, 150],
                'n_epochs': [10, 20, 30],
                'lr_all': [0.002, 0.005, 0.01],
                'reg_all': [0.02, 0.1, 0.4]
            }
            gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, joblib_verbose=0)
            gs.fit(data)
            
            logger.info(f"{algorithm_name} 最佳参数: {gs.best_params['rmse']}")
            logger.info(f"{algorithm_name} 最佳RMSE: {gs.best_score['rmse']:.4f}")
            
            # 更新算法
            self.algorithms[algorithm_name] = gs.best_estimator['rmse']
            
        return self.algorithms[algorithm_name]