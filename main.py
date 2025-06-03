import argparse
import sys
from core.recommendation_system import AdvancedRecommendationSystem
from utils.logger import setup_logger
from config.settings import Config

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='高级推荐系统')
    
    # 基本参数
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='运行模式: train(训练) 或 test(测试)')
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default=Config.DATASET_NAME,
                       help=f'数据集名称 (默认: {Config.DATASET_NAME})')
    parser.add_argument('--test-size', type=float, default=Config.TEST_SIZE,
                       help=f'测试集比例 (默认: {Config.TEST_SIZE})')
    
    # 模型参数
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['ncf', 'mf', 'neumf', 'fm','ncf_hybrid', 'mf_hybrid', 'neumf_hybrid', 'fm_hybrid'], 
                       default=['ncf'],
                       help='要训练的深度学习模型类型')
    parser.add_argument('--no-deep-learning', action='store_true',
                       help='禁用深度学习算法，只使用传统算法')
    
    # 深度学习超参数
    parser.add_argument('--embedding-dim', type=int, default=Config.EMBEDDING_DIM,
                       help=f'嵌入维度 (默认: {Config.EMBEDDING_DIM})')
    parser.add_argument('--hidden-dim', type=int, default=Config.HIDDEN_DIM,
                       help=f'隐藏层维度 (默认: {Config.HIDDEN_DIM})')
    parser.add_argument('--learning-rate', type=float, default=Config.LEARNING_RATE,
                       help=f'学习率 (默认: {Config.LEARNING_RATE})')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                       help=f'批大小 (默认: {Config.BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS,
                       help=f'训练轮数 (默认: {Config.EPOCHS})')
    # 添加autoencoder参数
    parser.add_argument('--autoencoder-hidden-dims', type=int, nargs='+', default=Config.AUTOENCODER_HIDDEN_DIMS,
                        help='AutoEncoder隐藏层维度列表，例如 --autoencoder-hidden-dims 64 32')
    parser.add_argument('--autoencoder-epochs', type=int, default=Config.AUTOENCODER_EPOCHS,
                        help='AutoEncoder训练轮数')
    # 添加聚类参数
    parser.add_argument('--n-clusters', type=int, default=Config.N_CLUSTERS,
                        help='用户聚类数量')
    
    # 模型保存/加载
    parser.add_argument('--model-dir', type=str, default=Config.MODEL_DIR,
                       help=f'模型保存目录 (默认: {Config.MODEL_DIR})')
    parser.add_argument('--model-path', type=str,
                       help='测试模式下的模型路径')
    
    # 日志参数
    parser.add_argument('--log-file', type=str,
                       help='指定日志文件名')
    parser.add_argument('--log-level', type=str, default=Config.LOG_LEVEL,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help=f'日志级别 (默认: {Config.LOG_LEVEL})')
    
    return parser.parse_args()

def update_config_from_args(args):
    """根据命令行参数更新配置"""
    Config.DATASET_NAME = args.dataset
    Config.TEST_SIZE = args.test_size
    Config.EMBEDDING_DIM = args.embedding_dim
    Config.HIDDEN_DIM = args.hidden_dim
    Config.LEARNING_RATE = args.learning_rate
    Config.BATCH_SIZE = args.batch_size
    Config.EPOCHS = args.epochs
    Config.MODEL_DIR = args.model_dir
    Config.LOG_LEVEL = args.log_level
    Config.AUTOENCODER_HIDDEN_DIMS = args.autoencoder_hidden_dims
    Config.AUTOENCODER_EPOCHS = args.autoencoder_epochs
    Config.N_CLUSTERS = args.n_clusters

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 更新配置
    update_config_from_args(args)
    
    # 设置日志
    logger = setup_logger(args.log_file, args)
    
    try:
        # 验证参数
        if args.mode == 'test' and not args.model_path:
            logger.error("测试模式需要指定 --model-path 参数")
            sys.exit(1)
        
        # 创建推荐系统实例
        use_deep_learning = not args.no_deep_learning
        rec_system = AdvancedRecommendationSystem(
            use_deep_learning=use_deep_learning,
            model_types=args.models,
            model_path=args.model_path,
            config=Config,
        )
        
        # 根据模式运行
        if args.mode == 'train':
            logger.info("开始训练模式")
            rec_system.train_mode()
        else:
            logger.info("开始测试模式")
            rec_system.test_mode()
            
        logger.info("程序执行完成")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()