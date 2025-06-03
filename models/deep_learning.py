import logging
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import collections
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm  # 进度条库
from utils.logger import setup_logger

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class GHRSFeatureExtractor:
    """GHRS特征提取器 - 基于surprise trainset"""
    
    def __init__(self, alpha_coefs=[0.02]):
        self.alpha_coefs = alpha_coefs
        self.user_features = None
        self.scaler = StandardScaler()
        
    def extract_features_from_trainset(self, trainset):
        """从surprise trainset提取GHRS特征"""
        logger.info("Extracting GHRS features from trainset...")
        
        # 从trainset转换为DataFrame
        ratings_data = []
        for uid, iid, rating in trainset.all_ratings():
            # trainset中的uid和iid是内部索引，需要转换为原始ID
            raw_uid = trainset.to_raw_uid(uid)
            raw_iid = trainset.to_raw_iid(iid)
            ratings_data.append([raw_uid, raw_iid, rating])
        
        df = pd.DataFrame(ratings_data, columns=['UID', 'MID', 'rate'])
        
        logger.info(f"Loaded {len(df)} ratings from trainset")
        logger.info(f"Users: {len(df['UID'].unique())}, Items: {len(df['MID'].unique())}")
        
        # 获取所有用户
        all_users = sorted(df['UID'].unique())
        num_items = len(df['MID'].unique())
        
        # 创建基础用户特征矩阵（由于没有人口统计学特征，我们创建基础特征）
        logger.info("Creating base user features...")
        df_user = pd.DataFrame({'UID': all_users})
        
        # 添加基于用户行为的基础特征
        user_stats = df.groupby('UID').agg({
            'rate': ['count', 'mean', 'std'],
            'MID': 'nunique'
        }).round(4)
        
        user_stats.columns = ['rating_count', 'rating_mean', 'rating_std', 'item_diversity']
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
        user_stats = user_stats.reset_index()
        
        # 合并用户统计特征
        df_user = df_user.merge(user_stats, on='UID', how='left')
        
        # 标准化统计特征
        feature_cols = ['rating_count', 'rating_mean', 'rating_std', 'item_diversity']
        df_user[feature_cols] = self.scaler.fit_transform(df_user[feature_cols])
        
        logger.info("Extracting graph features...")
        
        for alpha_coef in self.alpha_coefs:
            # 构建用户关系图
            pairs = []
            grouped = df.groupby(['MID', 'rate'])
            for key, group in grouped:
                user_list = group['UID'].tolist()
                pairs.extend(list(itertools.combinations(user_list, 2)))
            
            counter = collections.Counter(pairs)
            alpha = alpha_coef * num_items  # param * num_items
            edge_list = [list(el) for el in counter.keys() if counter[el] >= alpha]
            
            logger.info(f"Building graph with {len(edge_list)} edges (alpha={alpha_coef})")
            
            G = nx.Graph()
            # 确保所有用户都在图中
            G.add_nodes_from(all_users)
            
            for el in edge_list:
                G.add_edge(el[0], el[1], weight=1)
            
            # 添加自环
            for user in all_users:
                G.add_edge(user, user, weight=1)
            
            # 计算图特征
            logger.info(f"Computing graph centrality features for alpha={alpha_coef}...")
            
            # PageRank
            pr = nx.pagerank(G.to_directed())
            df_user['PR'] = df_user['UID'].map(pr).fillna(0)
            df_user['PR'] /= (df_user['PR'].max() + 1e-8)
            
            # Degree Centrality
            dc = nx.degree_centrality(G)
            df_user['CD'] = df_user['UID'].map(dc).fillna(0)
            df_user['CD'] /= (df_user['CD'].max() + 1e-8)
            
            # Closeness Centrality
            cc = nx.closeness_centrality(G)
            df_user['CC'] = df_user['UID'].map(cc).fillna(0)
            df_user['CC'] /= (df_user['CC'].max() + 1e-8)
            
            # Betweenness Centrality
            bc = nx.betweenness_centrality(G)
            df_user['CB'] = df_user['UID'].map(bc).fillna(0)
            df_user['CB'] /= (df_user['CB'].max() + 1e-8)
            
            # Load Centrality
            lc = nx.load_centrality(G)
            df_user['LC'] = df_user['UID'].map(lc).fillna(0)
            df_user['LC'] /= (df_user['LC'].max() + 1e-8)
            
            # Average Neighbor Degree
            nd = nx.average_neighbor_degree(G, weight='weight')
            df_user['AND'] = df_user['UID'].map(nd).fillna(0)
            df_user['AND'] /= (df_user['AND'].max() + 1e-8)
        
        # 填充缺失值
        X_train = df_user[df_user.columns[1:]]  # 排除UID列
        X_train.fillna(0, inplace=True)
        
        self.user_features = X_train
        return df_user, df, X_train

class AutoEncoder(nn.Module):
    """自动编码器用于特征降维"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super(AutoEncoder, self).__init__()
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器
        decoder_layers = []
        hidden_dims_reverse = hidden_dims[::-1]
        for i, hidden_dim in enumerate(hidden_dims_reverse):
            if i == len(hidden_dims_reverse) - 1:
                decoder_layers.append(nn.Linear(hidden_dim, input_dim))
            else:
                decoder_layers.extend([
                    nn.Linear(hidden_dim, hidden_dims_reverse[i+1]),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class NeuralCollaborativeFiltering(nn.Module):
    """神经协同过滤模型"""
    
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_dims=[128, 64]):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 拼接用户和物品嵌入
        x = torch.cat([user_emb, item_emb], dim=1)
        output = self.mlp(x)
        return output.squeeze()
    
class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorizationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        dot_product = torch.sum(user_emb * item_emb, dim=1)
        return dot_product
    
class NeuralMatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dims):
        super(NeuralMatrixFactorizationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        for i in range(1, len(hidden_dims)):
            self.mlp.add_module(f"linear_{i}", nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.mlp.add_module(f"relu_{i}", nn.ReLU())
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(0.2))
        
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 拼接用户和物品嵌入
        x = torch.cat([user_emb, item_emb], dim=1)
        
        mlp_output = self.mlp(x)
        output = self.output_layer(mlp_output)
        return output.squeeze()
    
class FactorizationMachineModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(FactorizationMachineModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.linear = nn.Linear(embedding_dim * 2, 1, bias=True)
        self.fm = nn.Linear(embedding_dim * 2, 1, bias=False)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 拼接用户和物品嵌入
        x = torch.cat([user_emb, item_emb], dim=1)
        
        linear_part = self.linear(x)
        fm_part = 0.5 * torch.sum(self.fm(x) ** 2 - self.fm(x ** 2), dim=1, keepdim=True)
        output = linear_part + fm_part
        return output.squeeze()

class HybridDeepModel(nn.Module):
    """混合深度模型，结合GHRS特征和协同过滤 - NCF的hybrid版本"""
    
    def __init__(self, num_users, num_items, user_feature_dim, 
                 embedding_dim=50, hidden_dims=[128, 64]):
        super(HybridDeepModel, self).__init__()
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 用户特征处理
        self.user_feature_mlp = nn.Sequential(
            nn.Linear(user_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 混合特征MLP
        mlp_layers = []
        input_dim = embedding_dim * 3  # user_emb + item_emb + user_features
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, user_ids, item_ids, user_features):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        user_feat_emb = self.user_feature_mlp(user_features)
        
        # 拼接所有特征
        x = torch.cat([user_emb, item_emb, user_feat_emb], dim=1)
        output = self.mlp(x)
        return output.squeeze()

class HybridMatrixFactorizationModel(nn.Module):
    """混合矩阵分解模型，结合GHRS特征"""
    
    def __init__(self, num_users, num_items, user_feature_dim, embedding_dim=50):
        super(HybridMatrixFactorizationModel, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 用户特征处理
        self.user_feature_mlp = nn.Sequential(
            nn.Linear(user_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 特征融合层
        self.fusion_layer = nn.Linear(embedding_dim * 2, embedding_dim)
        
    def forward(self, user_ids, item_ids, user_features):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        user_feat_emb = self.user_feature_mlp(user_features)
        
        # 融合用户嵌入和用户特征
        fused_user_emb = self.fusion_layer(torch.cat([user_emb, user_feat_emb], dim=1))
        
        # 计算点积
        dot_product = torch.sum(fused_user_emb * item_emb, dim=1)
        return dot_product

class HybridNeuralMatrixFactorizationModel(nn.Module):
    """混合神经矩阵分解模型，结合GHRS特征"""
    
    def __init__(self, num_users, num_items, user_feature_dim, embedding_dim=50, hidden_dims=[128, 64]):
        super(HybridNeuralMatrixFactorizationModel, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 用户特征处理
        self.user_feature_mlp = nn.Sequential(
            nn.Linear(user_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dims[0]),  # user_emb + item_emb + user_features
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        for i in range(1, len(hidden_dims)):
            self.mlp.add_module(f"linear_{i}", nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.mlp.add_module(f"relu_{i}", nn.ReLU())
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(0.2))
        
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, user_ids, item_ids, user_features):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        user_feat_emb = self.user_feature_mlp(user_features)
        
        # 拼接所有特征
        x = torch.cat([user_emb, item_emb, user_feat_emb], dim=1)
        
        mlp_output = self.mlp(x)
        output = self.output_layer(mlp_output)
        return output.squeeze()

class HybridFactorizationMachineModel(nn.Module):
    """混合分解机模型，结合GHRS特征"""
    
    def __init__(self, num_users, num_items, user_feature_dim, embedding_dim=50):
        super(HybridFactorizationMachineModel, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 用户特征处理
        self.user_feature_mlp = nn.Sequential(
            nn.Linear(user_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # FM层 - 处理拼接后的特征
        self.linear = nn.Linear(embedding_dim * 3, 1, bias=True)
        self.fm = nn.Linear(embedding_dim * 3, 1, bias=False)
        
    def forward(self, user_ids, item_ids, user_features):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        user_feat_emb = self.user_feature_mlp(user_features)
        
        # 拼接所有特征
        x = torch.cat([user_emb, item_emb, user_feat_emb], dim=1)
        
        linear_part = self.linear(x)
        fm_part = 0.5 * torch.sum(self.fm(x) ** 2 - self.fm(x ** 2), dim=1, keepdim=True)
        output = linear_part + fm_part
        return output.squeeze()

class MovieLensDataset(Dataset):
    """通用推荐数据集"""
    
    def __init__(self, user_ids, item_ids, ratings, user_features=None):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
        self.user_features = torch.FloatTensor(user_features) if user_features is not None else None
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        if self.user_features is not None:
            return (self.user_ids[idx], self.item_ids[idx], 
                   self.user_features[idx], self.ratings[idx])
        else:
            return (self.user_ids[idx], self.item_ids[idx], self.ratings[idx])

class DeepLearningRecommender:
    """深度学习推荐系统 - 支持从trainset训练和模型加载"""
    
    def __init__(self, model_path=None, model_type='ncf_hybrid', config=None):
        if config is None:
            from config.settings import Config
            self.config = Config
        else:
            self.config = config
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型和数据
        self.model = None
        self.autoencoder = None
        self.kmeans = None
        self.scaler = StandardScaler()
        
        # 特征提取器
        self.feature_extractor = GHRSFeatureExtractor()
        
        # 数据映射
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        
        # 如果提供了模型路径，则加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.info(f"Initialized DeepLearningRecommender with model_type: {model_type}")
            logger.info(f"Using device: {self.device}")
    
    def _prepare_data_from_trainset(self, trainset):
        """从surprise trainset准备训练数据"""
        logger.info("Extracting GHRS features from trainset...")
        df_user, df_rating, user_features = self.feature_extractor.extract_features_from_trainset(trainset)
        
        # 创建用户和物品映射
        unique_users = sorted(df_rating['UID'].unique())
        unique_items = sorted(df_rating['MID'].unique())
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # 转换ID
        df_rating['user_idx'] = df_rating['UID'].map(self.user_to_idx)
        df_rating['item_idx'] = df_rating['MID'].map(self.item_to_idx)
        
        # 准备用户特征矩阵
        user_feature_matrix = np.zeros((len(unique_users), user_features.shape[1]))
        for i, user_id in enumerate(unique_users):
            if user_id in df_user['UID'].values:
                user_row = df_user[df_user['UID'] == user_id].index[0]
                user_feature_matrix[i] = user_features.iloc[user_row].values
        
        return df_rating, user_feature_matrix, len(unique_users), len(unique_items)
    
    def _train_autoencoder(self, user_features, hidden_dims=None, epochs= None):
        """训练自动编码器进行特征降维"""
        if hidden_dims is None:
            hidden_dims = self.config.AUTOENCODER_HIDDEN_DIMS
        if epochs is None:
            epochs = self.config.AUTOENCODER_EPOCHS
        logger.info("Training AutoEncoder for feature reduction...")
        
        input_dim = user_features.shape[1]
        self.autoencoder = AutoEncoder(input_dim, hidden_dims).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        
        # 转换为张量
        X_tensor = torch.FloatTensor(user_features).to(self.device)
        
        for epoch in tqdm(range(epochs), desc='AutoEncoder Training'):
            optimizer.zero_grad()
            encoded, decoded = self.autoencoder(X_tensor)
            loss = criterion(decoded, X_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                logger.info(f"AutoEncoder Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # 获取降维后的特征
        self.autoencoder.eval()
        with torch.no_grad():
            reduced_features, _ = self.autoencoder(X_tensor)
            reduced_features = reduced_features.cpu().numpy()
        
        return reduced_features
    
    def _cluster_users(self, reduced_features, n_clusters=None):
        """对用户进行聚类"""
        if n_clusters is None:
            n_clusters = self.config.N_CLUSTERS
        logger.info(f"Clustering users into {n_clusters} clusters...")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.RANDOM_STATE)
        user_clusters = self.kmeans.fit_predict(reduced_features)
        
        return user_clusters
    
    def train(self, trainset, save_model=False, model_save_path=None, **kwargs):
        """训练模型"""
        logger.info("Starting training process...")
        
        # 从trainset准备数据
        df_rating, user_features, num_users, num_items = self._prepare_data_from_trainset(trainset)
        
        # GHRS特征处理（只有hybrid模型需要）
        if self.model_type.endswith('_hybrid'):
            # 训练自动编码器
            reduced_features = self._train_autoencoder(user_features)
            
            # 用户聚类
            user_clusters = self._cluster_users(reduced_features)
            self.user_clusters = user_clusters
            self.reduced_features = reduced_features
        
        # 准备训练数据
        train_data, test_data = train_test_split(df_rating, test_size=0.2, random_state=self.config.RANDOM_STATE)
        
        user_ids = train_data['user_idx'].values
        item_ids = train_data['item_idx'].values
        ratings = train_data['rate'].values
        
        # 构建模型
        if self.model_type == 'ncf':
            self.model = NeuralCollaborativeFiltering(
                num_users, num_items, 
                embedding_dim=kwargs.get('embedding_dim', self.config.EMBEDDING_DIM),
                hidden_dims=kwargs.get('hidden_dims', self.config.HIDDEN_DIMS)
            ).to(self.device)
            
            dataset = MovieLensDataset(user_ids, item_ids, ratings)
            
        elif self.model_type == 'ncf_hybrid':
            user_feat_for_training = user_features[user_ids]
            
            self.model = HybridDeepModel(
                num_users, num_items, user_features.shape[1],
                embedding_dim=kwargs.get('embedding_dim', self.config.EMBEDDING_DIM),
                hidden_dims=kwargs.get('hidden_dims', self.config.HIDDEN_DIMS)
            ).to(self.device)
            
            dataset = MovieLensDataset(user_ids, item_ids, ratings, user_feat_for_training)
            
        elif self.model_type == 'mf':
            self.model = MatrixFactorizationModel(
                    num_users, num_items, 
                    embedding_dim=kwargs.get('embedding_dim', self.config.EMBEDDING_DIM)
                ).to(self.device)
            dataset = MovieLensDataset(user_ids, item_ids, ratings)
            
        elif self.model_type == 'mf_hybrid':
            user_feat_for_training = user_features[user_ids]
            
            self.model = HybridMatrixFactorizationModel(
                num_users, num_items, user_features.shape[1],
                embedding_dim=kwargs.get('embedding_dim', self.config.EMBEDDING_DIM)
            ).to(self.device)
            
            dataset = MovieLensDataset(user_ids, item_ids, ratings, user_feat_for_training)
            
        elif self.model_type == 'neumf':
            self.model = NeuralMatrixFactorizationModel(
                    num_users, num_items,
                    embedding_dim=kwargs.get('embedding_dim', self.config.EMBEDDING_DIM),
                    hidden_dims=kwargs.get('hidden_dims', self.config.HIDDEN_DIMS)
                ).to(self.device)
            dataset = MovieLensDataset(user_ids, item_ids, ratings)
            
        elif self.model_type == 'neumf_hybrid':
            user_feat_for_training = user_features[user_ids]
            
            self.model = HybridNeuralMatrixFactorizationModel(
                num_users, num_items, user_features.shape[1],
                embedding_dim=kwargs.get('embedding_dim', self.config.EMBEDDING_DIM),
                hidden_dims=kwargs.get('hidden_dims', self.config.HIDDEN_DIMS)
            ).to(self.device)
            
            dataset = MovieLensDataset(user_ids, item_ids, ratings, user_feat_for_training)
            
        elif self.model_type == 'fm':
            self.model = FactorizationMachineModel(
                    num_users, num_items,
                    embedding_dim=kwargs.get('embedding_dim', self.config.EMBEDDING_DIM)
                ).to(self.device)
            dataset = MovieLensDataset(user_ids, item_ids, ratings)
            
        elif self.model_type == 'fm_hybrid':
            user_feat_for_training = user_features[user_ids]
            
            self.model = HybridFactorizationMachineModel(
                num_users, num_items, user_features.shape[1],
                embedding_dim=kwargs.get('embedding_dim', self.config.EMBEDDING_DIM)
            ).to(self.device)
            
            dataset = MovieLensDataset(user_ids, item_ids, ratings, user_feat_for_training)
        
        # 训练参数
        batch_size = kwargs.get('batch_size', self.config.BATCH_SIZE)
        epochs = kwargs.get('epochs', self.config.EPOCHS)
        learning_rate = kwargs.get('learning_rate', self.config.LEARNING_RATE)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练循环
        logger.info(f"Training {self.model_type} model...")
        self.model.train()
        
        for epoch in tqdm(range(epochs), desc='Model Training'):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                if self.model_type in ['ncf', 'mf', 'neumf', 'fm']:
                    user_batch, item_batch, rating_batch = batch
                    user_batch = user_batch.to(self.device)
                    item_batch = item_batch.to(self.device)
                    rating_batch = rating_batch.to(self.device)
                    
                    predictions = self.model(user_batch, item_batch)
                    
                elif self.model_type in ['ncf_hybrid', 'mf_hybrid', 'neumf_hybrid', 'fm_hybrid']:
                    user_batch, item_batch, user_feat_batch, rating_batch = batch
                    user_batch = user_batch.to(self.device)
                    item_batch = item_batch.to(self.device)
                    user_feat_batch = user_feat_batch.to(self.device)
                    rating_batch = rating_batch.to(self.device)
                    
                    predictions = self.model(user_batch, item_batch, user_feat_batch)
                    
                loss = criterion(predictions, rating_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        # 保存测试数据和用户特征用于评估和预测
        self.test_data = test_data
        self.user_features = user_features
        
        logger.info("Training completed!")
        
        # 保存模型
        if save_model:
            if model_save_path is None:
                model_save_path = self.config.MODEL_DIR +'/'+ self.config.MODEL_NAME_FORMAT.format(model_type=self.model_type, timestamp=pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'))
            self.save_model(model_save_path)
            return model_save_path
        
        return None
    
    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please call train() first or load a model.")
        
        # 转换ID
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return 3.0  # 默认评分
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            item_tensor = torch.LongTensor([item_idx]).to(self.device)
            
            if self.model_type in ['ncf', 'mf', 'neumf', 'fm']:
                prediction = self.model(user_tensor, item_tensor)
                
            elif self.model_type in ['ncf_hybrid', 'mf_hybrid', 'neumf_hybrid', 'fm_hybrid']:
                user_feat = torch.FloatTensor([self.user_features[user_idx]]).to(self.device)
                prediction = self.model(user_tensor, item_tensor, user_feat)
                
            return max(1.0, min(5.0, prediction.item()))  # 限制在1-5范围内
    
    def recommend(self, user_id, top_k=10):
        """为用户推荐top-k物品"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please call train() first or load a model.")
        
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # 获取所有物品的预测评分
        predictions = []
        for item_idx in range(len(self.idx_to_item)):
            item_id = self.idx_to_item[item_idx]
            score = self.predict(user_id, item_id)
            predictions.append((item_id, score))
        
        # 排序并返回top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]
    
    def save_model(self, path):
        """保存模型"""
        model_config = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_item': self.idx_to_item,
            'scaler': self.scaler,
            'feature_extractor_scaler': self.feature_extractor.scaler,
            'autoencoder': self.autoencoder.state_dict() if self.autoencoder else None,
            'kmeans': self.kmeans,
            'user_features': self.user_features,
            'user_feature_dim': self.user_features.shape[1] if hasattr(self, 'user_features') else None,
            'num_users': len(self.user_to_idx),
            'num_items': len(self.item_to_idx)
        }
        
        torch.save(model_config, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.user_to_idx = checkpoint['user_to_idx']
        self.item_to_idx = checkpoint['item_to_idx']
        self.idx_to_user = checkpoint['idx_to_user']
        self.idx_to_item = checkpoint['idx_to_item']
        self.scaler = checkpoint['scaler']
        self.feature_extractor.scaler = checkpoint['feature_extractor_scaler']
        self.kmeans = checkpoint['kmeans']
        self.user_features = checkpoint['user_features']
        
        # 重建模型架构
        num_users = checkpoint['num_users']
        num_items = checkpoint['num_items']
        user_feature_dim = checkpoint['user_feature_dim']
        
        if self.model_type == 'ncf':
            self.model = NeuralCollaborativeFiltering(num_users, num_items)
        elif self.model_type == 'ncf_hybrid':
            self.model = HybridDeepModel(num_users, num_items, user_feature_dim)
        elif self.model_type == 'mf':
            self.model = MatrixFactorizationModel(num_users, num_items, self.config.EMBEDDING_DIM)
        elif self.model_type == 'mf_hybrid':
            self.model = HybridMatrixFactorizationModel(num_users, num_items, user_feature_dim)
        elif self.model_type == 'neumf':
            self.model = NeuralMatrixFactorizationModel(num_users, num_items, self.config.EMBEDDING_DIM, self.config.HIDDEN_DIMS)
        elif self.model_type == 'neumf_hybrid':
            self.model = HybridNeuralMatrixFactorizationModel(num_users, num_items, user_feature_dim)
        elif self.model_type == 'fm':
            self.model = FactorizationMachineModel(num_users, num_items, self.config.EMBEDDING_DIM)
        elif self.model_type == 'fm_hybrid':
            self.model = HybridFactorizationMachineModel(num_users, num_items, user_feature_dim)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        if checkpoint['autoencoder']:
            self.autoencoder = AutoEncoder(user_feature_dim)
            self.autoencoder.load_state_dict(checkpoint['autoencoder'])
            self.autoencoder = self.autoencoder.to(self.device)
        
        logger.info(f"Model loaded from {path}")
    
    def evaluate(self):
        """评估模型性能"""
        if not hasattr(self, 'test_data'):
            logger.info("No test data available for evaluation.")
            return {}
        
        logger.info("Evaluating model performance...")
        
        predictions = []
        actuals = []
        
        for _, row in self.test_data.iterrows():
            user_id = row['UID']
            item_id = row['MID']
            actual_rating = row['rate']
            
            predicted_rating = self.predict(user_id, item_id)
            
            predictions.append(predicted_rating)
            actuals.append(actual_rating)
        
        # 计算指标
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        results = {
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse
        }
        
        logger.info("Evaluation Results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return results