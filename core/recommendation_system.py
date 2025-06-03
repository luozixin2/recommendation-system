# core/recommendation_system.py
import pandas as pd
import numpy as np
from surprise import Dataset, accuracy
from surprise.model_selection import train_test_split, cross_validate
from tqdm import tqdm
import time
import logging

from models.traditional import TraditionalRecommender
from models.deep_learning import DeepLearningRecommender
from utils.metrics import RecommendationMetrics
from utils.visualization import RecommendationVisualizer
from config.settings import Config

logger = logging.getLogger(__name__)

class AdvancedRecommendationSystem:
    """高级推荐系统主类"""
    
    def __init__(self, use_deep_learning=True, model_types=None, model_path=None, config=None):
        self.data = None
        self.trainset = None
        self.testset = None
        self.traditional_recommender = None
        self.deep_learning_recommender = None
        self.use_deep_learning = use_deep_learning
        self.model_types = model_types or ['ncf', 'mf', 'neumf', 'fm','ncf_hybrid', 'mf_hybrid', 'neumf_hybrid', 'fm_hybrid']
        self.model_path = model_path
        self.results = {}
        self.best_algorithm = None
        self.metrics = RecommendationMetrics()
        self.visualizer = RecommendationVisualizer()
        if config != None:
            self.config = config
        else:
            self.config = Config
        if model_path:
            # 如果指定了模型路径,则直接加载模型
            self.deep_learning_recommender = DeepLearningRecommender(model_path=model_path, config=config)
            self.use_deep_learning = True
        else:
            # 否则根据参数决定是否使用深度学习
            self.use_deep_learning = use_deep_learning
            if self.use_deep_learning:
                self.deep_learning_recommender = DeepLearningRecommender(config=config)
            else:
                self.traditional_recommender = TraditionalRecommender()
        logger.info("高级推荐系统初始化完成")
        
    def load_data(self, dataset_name=None):
        """加载数据集"""
        if dataset_name is None:
            dataset_name = Config.DATASET_NAME
            
        logger.info(f"开始加载数据集: {dataset_name}")
        
        with tqdm(total=1, desc="加载数据", leave=False) as pbar:
            self.data = Dataset.load_builtin(dataset_name)
            pbar.update(1)
        
        self._analyze_data()
        logger.info("数据加载和分析完成")
        
    def train_mode(self):
        """训练模式"""
        if self.deep_learning_recommender:
            # 如果有深度学习模型,则不再训练传统算法
            self.load_data()
            self.split_data()
            self._train_and_evaluate_deep_learning()
        else:
            # 否则按原有逻辑训练传统算法和深度学习算法
            self.load_data()
            self.split_data()
            self.train_and_evaluate()
        
    def test_mode(self):
        """测试模式"""
        if not self.model_path:
            raise ValueError("测试模式需要指定模型路径")
            
        logger.info("进入测试模式")
        self.load_data()
        self.split_data()
        
        # 加载指定模型进行测试
        dl_recommender = DeepLearningRecommender(model_path=self.model_path, config=self.config)
        
        # 在测试集上评估
        predictions = []
        true_ratings = []
        
        logger.info("开始在测试集上评估模型")
        with tqdm(total=len(self.testset), desc="模型测试") as pbar:
            for user, item, true_rating in self.testset:
                pred_rating = dl_recommender.predict(user, item)
                predictions.append(pred_rating)
                true_ratings.append(true_rating)
                pbar.update(1)
        
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(true_ratings)) ** 2))
        mae = np.mean(np.abs(np.array(predictions) - np.array(true_ratings)))
        
        print(f"\n=== 模型测试结果 ===")
        print(f"模型路径: {self.model_path}")
        print(f"测试集 RMSE: {rmse:.4f}")
        print(f"测试集 MAE: {mae:.4f}")
        
        logger.info(f"模型测试完成 - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
    def train_and_evaluate(self):
        """训练和评估所有算法"""
        print("\n=== 训练和评估算法 ===")
        logger.info("开始训练和评估所有算法")
        
        # 传统算法
        traditional_algorithms = self.traditional_recommender.get_algorithms()
        total_algorithms = len(traditional_algorithms)
        
        if self.use_deep_learning:
            total_algorithms += len(self.model_types)
        
        with tqdm(total=total_algorithms, desc="算法训练进度") as main_pbar:
            # 训练传统算法
            for name, algorithm in traditional_algorithms.items():
                self._train_traditional_algorithm(name, algorithm)
                main_pbar.update(1)
            
            # 训练深度学习算法
            if self.use_deep_learning:
                for model_type in self.model_types:
                    saved_path = self._train_deep_learning_algorithm(model_type)
                    if saved_path:
                        logger.info(f"{model_type} 模型已保存到: {saved_path}")
                    main_pbar.update(1)
        
        self._display_results()
        
    def _train_deep_learning_algorithm(self, model_type):
        """训练深度学习算法"""
        name = f"DeepLearning_{model_type.upper()}"
        logger.info(f"开始训练深度学习算法: {name}")
        
        try:
            dl_recommender = DeepLearningRecommender(model_type=model_type,config=self.config)
            saved_path = dl_recommender.train(self.trainset, save_model=True)
            
            # 在测试集上评估
            predictions = []
            true_ratings = []
            
            for user, item, true_rating in self.testset:
                pred_rating = dl_recommender.predict(user, item)
                predictions.append(pred_rating)
                true_ratings.append(true_rating)
            
            rmse = np.sqrt(np.mean((np.array(predictions) - np.array(true_ratings)) ** 2))
            mae = np.mean(np.abs(np.array(predictions) - np.array(true_ratings)))
            
            self.results[name] = {
                'type': 'deep_learning',
                'cv_rmse_mean': rmse,
                'cv_rmse_std': 0,
                'cv_mae_mean': mae,
                'cv_mae_std': 0,
                'test_rmse': rmse,
                'test_mae': mae,
                'algorithm': dl_recommender,
                'predictions': list(zip(predictions, true_ratings)),
                'model_path': saved_path
            }
            
            logger.info(f"{name} - Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")
            return saved_path
            
        except Exception as e:
            logger.exception(f"训练 {name} 时出错: {str(e)}")
            return None
        
        
    def _analyze_data(self):
        """数据分析"""
        print("\n=== 数据集分析 ===")
        
        with tqdm(total=3, desc="数据分析", leave=False) as pbar:
            df = pd.DataFrame(self.data.raw_ratings, columns=['user', 'item', 'rating', 'timestamp'])
            pbar.update(1)
            
            stats = self.metrics.calculate_dataset_stats(df)
            pbar.update(1)
            
            for key, value in stats.items():
                print(f"{key}: {value}")
                logger.info(f"{key}: {value}")
            
            # 可视化
            self.visualizer.plot_data_analysis(df)
            pbar.update(1)
            
    def split_data(self, test_size=None):
        """分割训练集和测试集"""
        if test_size is None:
            test_size = Config.TEST_SIZE
            
        logger.info(f"开始分割数据集 (测试集比例: {test_size})")
        
        with tqdm(total=1, desc="数据分割", leave=False) as pbar:
            self.trainset, self.testset = train_test_split(
                self.data, test_size=test_size, random_state=Config.RANDOM_STATE
            )
            pbar.update(1)
        
        train_info = f"训练集: {self.trainset.n_users} 用户, {self.trainset.n_items} 物品"
        test_info = f"测试集: {len(self.testset)} 条评分记录"
        
        print(f"\n{train_info}")
        print(f"{test_info}")
        logger.info(train_info)
        logger.info(test_info)
        
    def train_and_evaluate(self):
        """训练和评估所有算法"""
        print("\n=== 训练和评估算法 ===")
        logger.info("开始训练和评估所有算法")
        
        # 传统算法
        traditional_algorithms = self.traditional_recommender.get_algorithms()
        total_algorithms = len(traditional_algorithms)
        
        if self.use_deep_learning:
            total_algorithms += 3  # NCF, AutoEncoder, VAE
        
        with tqdm(total=total_algorithms, desc="算法训练进度") as main_pbar:
            # 训练传统算法
            for name, algorithm in traditional_algorithms.items():
                self._train_traditional_algorithm(name, algorithm)
                main_pbar.update(1)
            
            # 训练深度学习算法
            if self.use_deep_learning:
                for model_type in ['ncf', 'autoencoder', 'vae']:
                    self._train_deep_learning_algorithm(model_type)
                    main_pbar.update(1)
        
        self._display_results()
        
    def _train_traditional_algorithm(self, name, algorithm):
        """训练传统算法"""
        logger.info(f"开始训练算法: {name}")
        
        # 交叉验证
        cv_results = cross_validate(algorithm, self.data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        
        # 在测试集上评估
        algorithm.fit(self.trainset)
        predictions = algorithm.test(self.testset)
        
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        self.results[name] = {
            'type': 'traditional',
            'cv_rmse_mean': cv_results['test_rmse'].mean(),
            'cv_rmse_std': cv_results['test_rmse'].std(),
            'cv_mae_mean': cv_results['test_mae'].mean(),
            'cv_mae_std': cv_results['test_mae'].std(),
            'test_rmse': rmse,
            'test_mae': mae,
            'algorithm': algorithm,
            'predictions': predictions
        }
        
        logger.info(f"{name} - Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")
        
    def _train_and_evaluate_deep_learning(self):
        """训练和评估深度学习算法"""
        for model_type in self.model_types:
            self._train_deep_learning_algorithm(model_type)

        self._display_results()


    def _evaluate_deep_learning_model(self, recommender):
        """在测试集上评估深度学习模型"""
        predictions = []
        true_ratings = []

        for user, item, true_rating in self.testset:
            pred_rating = recommender.predict(user, item)
            predictions.append(pred_rating)
            true_ratings.append(true_rating)

        return predictions, true_ratings
    
    def _display_results(self):
        """显示评估结果"""
        print("\n=== 算法性能比较 ===")
        logger.info("算法性能比较结果:")
        
        results_df = pd.DataFrame({
            name: {
                'Type': result['type'],
                'CV RMSE': f"{result['cv_rmse_mean']:.4f} ± {result['cv_rmse_std']:.4f}",
                'CV MAE': f"{result['cv_mae_mean']:.4f} ± {result['cv_mae_std']:.4f}",
                'Test RMSE': f"{result['test_rmse']:.4f}",
                'Test MAE': f"{result['test_mae']:.4f}"
            }
            for name, result in self.results.items()
        }).T
        
        print(results_df)
        
        # 找出最佳算法
        best_rmse = min(self.results.items(), key=lambda x: x[1]['test_rmse'])
        self.best_algorithm = best_rmse[0]
        best_info = f"最佳算法: {self.best_algorithm} (RMSE: {best_rmse[1]['test_rmse']:.4f})"
        
        print(f"\n{best_info}")
        logger.info(best_info)
        
        # 可视化结果
        self.visualizer.plot_algorithm_comparison(self.results)
        
    def run_complete_analysis(self):
        """运行完整分析"""
        print("开始运行完整的推荐系统分析...")
        logger.info("="*50)
        logger.info("开始运行完整的推荐系统分析")
        logger.info("="*50)
        
        steps = [
            ("加载和分析数据", self.load_data),
            ("分割数据", self.split_data),
            ("训练和评估", self.train_and_evaluate),
        ]
        
        with tqdm(total=len(steps), desc="总体分析进度") as main_pbar:
            for step_name, step_func in steps:
                print(f"\n{'='*20} {step_name} {'='*20}")
                start_time = time.time()
                
                step_func()
                    
                end_time = time.time()
                duration = end_time - start_time
                
                logger.info(f"{step_name} 完成，耗时: {duration:.2f}秒")
                main_pbar.update(1)
        
        print("\n=== 分析完成 ===")
        logger.info("="*50)
        logger.info("完整分析结束")
        logger.info("="*50)