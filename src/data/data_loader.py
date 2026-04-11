"""
数据加载和预处理模块
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    数据加载器
    负责加载Parquet文件并进行基础处理
    """
    
    def __init__(self, data_path: str):
        """
        初始化数据加载器
        
        Args:
            data_path: Parquet文件路径
        """
        self.data_path = data_path
        self.df = None
        self.feature_cols = None
        self.label_col = 'label_type'
        
    def load(self) -> pd.DataFrame:
        """
        加载数据
        
        Returns:
            DataFrame
        """
        print(f"正在加载数据: {self.data_path}")
        self.df = pd.read_parquet(self.data_path)
        print(f"数据加载完成，形状: {self.df.shape}")
        return self.df
    
    def get_feature_columns(self) -> Dict[str, List[str]]:
        """
        获取特征列分类
        
        Returns:
            特征列分类字典
        """
        if self.df is None:
            raise ValueError("请先调用 load() 加载数据")
        
        feature_dict = {
            'id_label': ['user_id', 'item_id', 'label_type', 'label_time', 'timestamp'],
            'user_int_scalar': [],
            'user_int_array': [],
            'user_dense': [],
            'item_int_scalar': [],
            'item_int_array': [],
            'domain_a': [],
            'domain_b': [],
            'domain_c': [],
            'domain_d': []
        }
        
        for col in self.df.columns:
            if col.startswith('user_int_feats_'):
                if self.df[col].dtype == 'object':
                    feature_dict['user_int_array'].append(col)
                else:
                    feature_dict['user_int_scalar'].append(col)
            elif col.startswith('user_dense_feats_'):
                feature_dict['user_dense'].append(col)
            elif col.startswith('item_int_feats_'):
                if self.df[col].dtype == 'object':
                    feature_dict['item_int_array'].append(col)
                else:
                    feature_dict['item_int_scalar'].append(col)
            elif col.startswith('domain_a_seq_'):
                feature_dict['domain_a'].append(col)
            elif col.startswith('domain_b_seq_'):
                feature_dict['domain_b'].append(col)
            elif col.startswith('domain_c_seq_'):
                feature_dict['domain_c'].append(col)
            elif col.startswith('domain_d_seq_'):
                feature_dict['domain_d'].append(col)
        
        return feature_dict
    
    def get_basic_info(self) -> Dict[str, Any]:
        """
        获取数据基本信息
        
        Returns:
            基本信息字典
        """
        if self.df is None:
            raise ValueError("请先调用 load() 加载数据")
        
        info = {
            'n_samples': len(self.df),
            'n_features': len(self.df.columns),
            'n_users': self.df['user_id'].nunique(),
            'n_items': self.df['item_id'].nunique(),
            'label_distribution': self.df['label_type'].value_counts().to_dict(),
            'missing_cols': self.df.isnull().sum()[self.df.isnull().sum() > 0].to_dict()
        }
        
        return info


class FeatureProcessor:
    """
    特征处理器
    负责特征工程和数据预处理
    """
    
    def __init__(self):
        """初始化特征处理器"""
        self.scalers = {}
        self.label_encoders = {}
        self.feature_stats = {}
        
    def process_array_features(self, df: pd.DataFrame, array_cols: List[str],
                               aggregation: str = 'mean') -> pd.DataFrame:
        """
        处理数组类型特征
        
        Args:
            df: 输入数据
            array_cols: 数组类型列名
            aggregation: 聚合方式 ('mean', 'max', 'min', 'sum', 'len')
            
        Returns:
            处理后的DataFrame
        """
        df_processed = df.copy()
        
        for col in array_cols:
            if col not in df.columns:
                continue
                
            if aggregation == 'mean':
                df_processed[f'{col}_mean'] = df[col].apply(
                    lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
                )
            elif aggregation == 'max':
                df_processed[f'{col}_max'] = df[col].apply(
                    lambda x: np.max(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
                )
            elif aggregation == 'min':
                df_processed[f'{col}_min'] = df[col].apply(
                    lambda x: np.min(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
                )
            elif aggregation == 'sum':
                df_processed[f'{col}_sum'] = df[col].apply(
                    lambda x: np.sum(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
                )
            elif aggregation == 'len':
                df_processed[f'{col}_len'] = df[col].apply(
                    lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
                )
            
            # 删除原始数组列
            df_processed = df_processed.drop(columns=[col])
        
        return df_processed
    
    def process_sequence_features(self, df: pd.DataFrame, seq_cols: List[str],
                                  max_len: int = 50, padding: str = 'post') -> pd.DataFrame:
        """
        处理序列特征
        
        Args:
            df: 输入数据
            seq_cols: 序列特征列名
            max_len: 最大序列长度
            padding: 填充方式 ('post' 或 'pre')
            
        Returns:
            处理后的DataFrame
        """
        df_processed = df.copy()
        
        for col in seq_cols:
            if col not in df.columns:
                continue
            
            # 序列长度统计
            df_processed[f'{col}_len'] = df[col].apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
            )
            
            # 序列聚合特征
            df_processed[f'{col}_mean'] = df[col].apply(
                lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
            )
            
            df_processed[f'{col}_max'] = df[col].apply(
                lambda x: np.max(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
            )
            
            # 删除原始序列列（简化处理，实际可能需要保留用于深度学习）
            df_processed = df_processed.drop(columns=[col])
        
        return df_processed
    
    def normalize_features(self, df: pd.DataFrame, numeric_cols: List[str],
                          fit: bool = True) -> pd.DataFrame:
        """
        标准化数值特征
        
        Args:
            df: 输入数据
            numeric_cols: 数值列名
            fit: 是否拟合新的scaler
            
        Returns:
            标准化后的DataFrame
        """
        df_processed = df.copy()
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            if fit:
                scaler = StandardScaler()
                df_processed[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                if col in self.scalers:
                    df_processed[col] = self.scalers[col].transform(df[[col]])
        
        return df_processed
    
    def encode_categorical(self, df: pd.DataFrame, cat_cols: List[str],
                          fit: bool = True) -> pd.DataFrame:
        """
        编码分类特征
        
        Args:
            df: 输入数据
            cat_cols: 分类列名
            fit: 是否拟合新的encoder
            
        Returns:
            编码后的DataFrame
        """
        df_processed = df.copy()
        
        for col in cat_cols:
            if col not in df.columns:
                continue
            
            if fit:
                encoder = LabelEncoder()
                df_processed[col] = encoder.fit_transform(df[col].astype(str))
                self.label_encoders[col] = encoder
            else:
                if col in self.label_encoders:
                    df_processed[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df_processed
    
    def prepare_features_for_ml(self, df: pd.DataFrame,
                                feature_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """
        为传统机器学习模型准备特征
        
        Args:
            df: 输入数据
            feature_dict: 特征分类字典
            
        Returns:
            处理后的DataFrame
        """
        df_processed = df.copy()
        
        # 处理数组特征 - 使用均值池化
        all_array_cols = (feature_dict['user_int_array'] + 
                         feature_dict['user_dense'] + 
                         feature_dict['item_int_array'])
        df_processed = self.process_array_features(df_processed, all_array_cols, 'mean')
        
        # 处理序列特征 - 使用统计聚合
        all_seq_cols = (feature_dict['domain_a'] + 
                       feature_dict['domain_b'] + 
                       feature_dict['domain_c'] + 
                       feature_dict['domain_d'])
        df_processed = self.process_sequence_features(df_processed, all_seq_cols)
        
        # 处理ID列 - 作为分类特征
        id_cols = ['user_id', 'item_id']
        df_processed = self.encode_categorical(df_processed, id_cols, fit=True)
        
        # 处理时间戳
        if 'label_time' in df_processed.columns:
            df_processed['label_time_hour'] = pd.to_datetime(
                df_processed['label_time'], unit='s'
            ).dt.hour
            df_processed['label_time_dayofweek'] = pd.to_datetime(
                df_processed['label_time'], unit='s'
            ).dt.dayofweek
        
        # 删除不需要的列
        cols_to_drop = ['timestamp']
        df_processed = df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns])
        
        # 填充缺失值（用0填充）
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0)
        
        return df_processed


def split_data(df: pd.DataFrame,
               test_size: float = 0.2,
               val_size: float = 0.1,
               random_state: int = 42,
               stratify_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    划分训练集、验证集、测试集
    
    Args:
        df: 输入数据
        test_size: 测试集比例
        val_size: 验证集比例（相对于训练集）
        random_state: 随机种子
        stratify_col: 分层抽样列
        
    Returns:
        (train_df, val_df, test_df)
    """
    stratify = df[stratify_col] if stratify_col else None
    
    # 先划分出测试集
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    # 再划分出验证集
    stratify_val = train_val_df[stratify_col] if stratify_col else None
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, random_state=random_state, stratify=stratify_val
    )
    
    print(f"数据划分完成:")
    print(f"  训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 条 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.1f}%)")
    
    if stratify_col:
        print(f"\n训练集 {stratify_col} 分布:")
        print(train_df[stratify_col].value_counts(normalize=True))
        print(f"\n验证集 {stratify_col} 分布:")
        print(val_df[stratify_col].value_counts(normalize=True))
        print(f"\n测试集 {stratify_col} 分布:")
        print(test_df[stratify_col].value_counts(normalize=True))
    
    return train_df, val_df, test_df
