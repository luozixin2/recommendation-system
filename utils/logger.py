import logging
import os
import json
from config.settings import Config

def setup_logger(log_file=None, args=None):
    """设置日志记录器"""
    if log_file is None:
        log_file = Config.get_log_filename()
    
    # 创建logs目录
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    log_path = os.path.join(Config.LOG_DIR, log_file)
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_path}")
    
    # 记录用户传入的参数
    if args:
        logger.info("="*50)
        logger.info("用户传入参数:")
        for key, value in vars(args).items():
            logger.info(f"  {key}: {value}")
        logger.info("="*50)
    
    return logger