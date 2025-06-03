# config/settings.py
import os
from datetime import datetime

class Config:
    # 数据配置
    DATASET_NAME = 'ml-100k'
    TEST_SIZE = 0.2
    
    # 模型配置
    RANDOM_STATE = 42
    N_RECOMMENDATIONS = 10
    
    # 深度学习配置
    EMBEDDING_DIM = 64
    EMBEDDING_DIMS = [128, 64]
    HIDDEN_DIM = 128
    HIDDEN_DIMS = [128, 64]
    LEARNING_RATE = 0.001
    BATCH_SIZE = 1024
    EPOCHS = 50
    
    # AutoEncoder参数
    AUTOENCODER_HIDDEN_DIMS = [64, 32]
    AUTOENCODER_EPOCHS = 100
    
    # KMeans参数
    N_CLUSTERS = 10
    
    # 模型保存配置
    MODEL_DIR = 'models/saved'
    MODEL_NAME_FORMAT = "{model_type}_{timestamp}.pth"
    
    # 日志配置
    LOG_DIR = 'logs'
    LOG_LEVEL = 'INFO'
    
    # 可视化配置
    FIGURE_SIZE = (12, 6)
    DPI = 100
    
    @staticmethod
    def get_log_filename():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"recommendation_system_{timestamp}.log"
    
    @staticmethod
    def get_model_filename(model_type):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return Config.MODEL_NAME_FORMAT.format(model_type=model_type, timestamp=timestamp)
    
    @staticmethod
    def ensure_directories():
        """确保必要的目录存在"""
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)