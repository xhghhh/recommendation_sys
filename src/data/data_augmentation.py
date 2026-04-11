"""
数据增强模块
用于扩充训练数据，缓解过拟合
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import random
from copy import deepcopy


class DataAugmentor:
    """
    数据增强器
    支持标量、数组、序列特征的多种增强策略
    """
    
    def __init__(self, random_state: int = 42):
        """
        初始化增强器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
    def augment_scalar_gaussian_noise(self, df: pd.DataFrame, 
                                      scalar_cols: List[str],
                                      sigma_ratio: float = 0.01) -> pd.DataFrame:
        """
        对标量特征添加高斯噪声
        
        Args:
            df: 输入数据
            scalar_cols: 标量特征列名
            sigma_ratio: 噪声标准差相对于特征标准差的比例
            
        Returns:
            增强后的数据
        """
        df_aug = df.copy()
        
        for col in scalar_cols:
            if col not in df.columns:
                continue
                
            # 计算特征标准差
            feature_std = df[col].std()
            if feature_std == 0 or np.isnan(feature_std):
                continue
                
            # 生成噪声
            noise = np.random.normal(0, sigma_ratio * feature_std, size=len(df))
            df_aug[col] = df[col] + noise
            
        return df_aug
    
    def augment_scalar_random_scale(self, df: pd.DataFrame,
                                    scalar_cols: List[str],
                                    scale_range: Tuple[float, float] = (0.95, 1.05)) -> pd.DataFrame:
        """
        对标量特征进行随机缩放
        
        Args:
            df: 输入数据
            scalar_cols: 标量特征列名
            scale_range: 缩放范围
            
        Returns:
            增强后的数据
        """
        df_aug = df.copy()
        
        for col in scalar_cols:
            if col not in df.columns:
                continue
                
            # 为每个样本生成随机缩放因子
            scales = np.random.uniform(scale_range[0], scale_range[1], size=len(df))
            df_aug[col] = df[col] * scales
            
        return df_aug
    
    def augment_array_random_mask(self, df: pd.DataFrame,
                                  array_cols: List[str],
                                  mask_ratio: float = 0.1) -> pd.DataFrame:
        """
        对数组特征进行随机mask
        
        Args:
            df: 输入数据
            array_cols: 数组特征列名
            mask_ratio: mask比例
            
        Returns:
            增强后的数据
        """
        df_aug = df.copy()
        
        for col in array_cols:
            if col not in df.columns:
                continue
                
            augmented_arrays = []
            for arr in df[col]:
                if not isinstance(arr, (list, np.ndarray)) or len(arr) == 0:
                    augmented_arrays.append(arr)
                    continue
                    
                # 随机mask
                arr = np.array(arr)
                mask = np.random.random(arr.shape) > mask_ratio
                augmented_arr = arr * mask
                augmented_arrays.append(augmented_arr.tolist())
                
            df_aug[col] = augmented_arrays
            
        return df_aug
    
    def augment_array_random_crop(self, df: pd.DataFrame,
                                  array_cols: List[str],
                                  crop_ratio: float = 0.8) -> pd.DataFrame:
        """
        对数组特征进行随机裁剪
        
        Args:
            df: 输入数据
            array_cols: 数组特征列名
            crop_ratio: 保留比例
            
        Returns:
            增强后的数据
        """
        df_aug = df.copy()
        
        for col in array_cols:
            if col not in df.columns:
                continue
                
            augmented_arrays = []
            for arr in df[col]:
                if not isinstance(arr, (list, np.ndarray)) or len(arr) == 0:
                    augmented_arrays.append(arr)
                    continue
                    
                arr = list(arr)
                if len(arr) <= 2:
                    augmented_arrays.append(arr)
                    continue
                    
                # 计算裁剪长度
                crop_length = max(1, int(len(arr) * crop_ratio))
                start = random.randint(0, len(arr) - crop_length)
                augmented_arr = arr[start:start + crop_length]
                augmented_arrays.append(augmented_arr)
                
            df_aug[col] = augmented_arrays
            
        return df_aug
    
    def augment_sequence_random_mask(self, df: pd.DataFrame,
                                     seq_cols: List[str],
                                     mask_ratio: float = 0.15) -> pd.DataFrame:
        """
        对序列特征进行随机mask
        
        Args:
            df: 输入数据
            seq_cols: 序列特征列名
            mask_ratio: mask比例
            
        Returns:
            增强后的数据
        """
        df_aug = df.copy()
        
        for col in seq_cols:
            if col not in df.columns:
                continue
                
            augmented_seqs = []
            for seq in df[col]:
                if not isinstance(seq, (list, np.ndarray)) or len(seq) == 0:
                    augmented_seqs.append(seq)
                    continue
                    
                seq = np.array(seq)
                if len(seq) <= 2:
                    augmented_seqs.append(seq.tolist())
                    continue
                    
                # 随机选择mask的位置
                n_mask = max(1, int(len(seq) * mask_ratio))
                mask_indices = random.sample(range(len(seq)), n_mask)
                
                augmented_seq = seq.copy()
                for idx in mask_indices:
                    augmented_seq[idx] = 0  # 用0填充
                    
                augmented_seqs.append(augmented_seq.tolist())
                
            df_aug[col] = augmented_seqs
            
        return df_aug
    
    def augment_sequence_subsample(self, df: pd.DataFrame,
                                   seq_cols: List[str],
                                   subsample_ratio: float = 0.8) -> pd.DataFrame:
        """
        对序列特征进行子序列采样
        
        Args:
            df: 输入数据
            seq_cols: 序列特征列名
            subsample_ratio: 采样比例
            
        Returns:
            增强后的数据
        """
        df_aug = df.copy()
        
        for col in seq_cols:
            if col not in df.columns:
                continue
                
            augmented_seqs = []
            for seq in df[col]:
                if not isinstance(seq, (list, np.ndarray)) or len(seq) == 0:
                    augmented_seqs.append(seq)
                    continue
                    
                seq = list(seq)
                if len(seq) <= 3:
                    augmented_seqs.append(seq)
                    continue
                    
                # 计算子序列长度
                subseq_length = max(2, int(len(seq) * subsample_ratio))
                start = random.randint(0, len(seq) - subseq_length)
                augmented_seq = seq[start:start + subseq_length]
                augmented_seqs.append(augmented_seq)
                
            df_aug[col] = augmented_seqs
            
        return df_aug
    
    def augment_combined_conservative(self, df: pd.DataFrame,
                                      scalar_cols: List[str],
                                      array_cols: List[str],
                                      seq_cols: List[str]) -> pd.DataFrame:
        """
        保守增强方案（推荐）
        
        Args:
            df: 输入数据
            scalar_cols: 标量特征列
            array_cols: 数组特征列
            seq_cols: 序列特征列
            
        Returns:
            增强后的数据
        """
        # 标量: 高斯噪声
        df_aug = self.augment_scalar_gaussian_noise(df, scalar_cols, sigma_ratio=0.01)
        
        # 数组: 随机mask
        df_aug = self.augment_array_random_mask(df_aug, array_cols, mask_ratio=0.1)
        
        # 序列: mask + 子序列采样
        df_aug = self.augment_sequence_random_mask(df_aug, seq_cols, mask_ratio=0.1)
        df_aug = self.augment_sequence_subsample(df_aug, seq_cols, subsample_ratio=0.8)
        
        return df_aug
    
    def augment_combined_aggressive(self, df: pd.DataFrame,
                                    scalar_cols: List[str],
                                    array_cols: List[str],
                                    seq_cols: List[str]) -> pd.DataFrame:
        """
        激进增强方案
        
        Args:
            df: 输入数据
            scalar_cols: 标量特征列
            array_cols: 数组特征列
            seq_cols: 序列特征列
            
        Returns:
            增强后的数据
        """
        # 标量: 噪声 + 缩放
        df_aug = self.augment_scalar_gaussian_noise(df, scalar_cols, sigma_ratio=0.05)
        df_aug = self.augment_scalar_random_scale(df_aug, scalar_cols, scale_range=(0.9, 1.1))
        
        # 数组: mask + 裁剪
        df_aug = self.augment_array_random_mask(df_aug, array_cols, mask_ratio=0.2)
        df_aug = self.augment_array_random_crop(df_aug, array_cols, crop_ratio=0.8)
        
        # 序列: mask + 子序列采样
        df_aug = self.augment_sequence_random_mask(df_aug, seq_cols, mask_ratio=0.15)
        df_aug = self.augment_sequence_subsample(df_aug, seq_cols, subsample_ratio=0.7)
        
        return df_aug
    
    def generate_augmented_dataset(self, df: pd.DataFrame,
                                   scalar_cols: List[str],
                                   array_cols: List[str],
                                   seq_cols: List[str],
                                   augmentation_factor: int = 2,
                                   strategy: str = 'conservative') -> pd.DataFrame:
        """
        生成增强后的数据集
        
        Args:
            df: 原始数据
            scalar_cols: 标量特征列
            array_cols: 数组特征列
            seq_cols: 序列特征列
            augmentation_factor: 增强倍数
            strategy: 增强策略 ('conservative' 或 'aggressive')
            
        Returns:
            原始数据 + 增强数据
        """
        augmented_dfs = [df]  # 保留原始数据
        
        for i in range(augmentation_factor - 1):
            # 设置不同的随机种子
            np.random.seed(self.random_state + i + 1)
            random.seed(self.random_state + i + 1)
            
            if strategy == 'conservative':
                aug_df = self.augment_combined_conservative(df, scalar_cols, array_cols, seq_cols)
            elif strategy == 'aggressive':
                aug_df = self.augment_combined_aggressive(df, scalar_cols, array_cols, seq_cols)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            augmented_dfs.append(aug_df)
        
        # 合并所有数据
        final_df = pd.concat(augmented_dfs, ignore_index=True)
        
        print(f"数据增强完成:")
        print(f"  原始样本数: {len(df)}")
        print(f"  增强后样本数: {len(final_df)}")
        print(f"  增强倍数: {augmentation_factor}x")
        print(f"  策略: {strategy}")
        
        return final_df


class SMOTEAugmentor:
    """
    SMOTE 过采样增强器
    用于解决类别不平衡问题
    """
    
    def __init__(self, random_state: int = 42):
        """
        初始化
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def fit_resample(self, X: pd.DataFrame, y: pd.Series,
                     sampling_strategy: float = 1.0) -> Tuple[pd.DataFrame, pd.Series]:
        """
        使用SMOTE进行过采样
        
        Args:
            X: 特征数据
            y: 标签数据
            sampling_strategy: 采样策略，1.0表示少数类与多数类样本数相同
            
        Returns:
            增强后的X, y
        """
        try:
            from imblearn.over_sampling import SMOTE
            
            smote = SMOTE(sampling_strategy=sampling_strategy,
                         random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            print(f"SMOTE过采样完成:")
            print(f"  原始分布: {y.value_counts().to_dict()}")
            print(f"  采样后分布: {pd.Series(y_resampled).value_counts().to_dict()}")
            
            return X_resampled, y_resampled
            
        except ImportError:
            print("警告: imblearn 未安装，跳过SMOTE。请运行: pip install imbalanced-learn")
            return X, y


class MixupAugmentor:
    """
    Mixup 数据增强
    用于深度学习训练
    """
    
    def __init__(self, alpha: float = 0.2, random_state: int = 42):
        """
        初始化
        
        Args:
            alpha: Beta分布参数，越小混合越极端
            random_state: 随机种子
        """
        self.alpha = alpha
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_mixup(self, X: np.ndarray, y: np.ndarray,
                       batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成Mixup增强的batch
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,)
            batch_size: batch大小
            
        Returns:
            混合后的X, y
        """
        # 随机选择两个样本
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # 采样混合系数
        lam = np.random.beta(self.alpha, self.alpha, size=len(X))
        
        # 混合特征
        X_mixed = lam.reshape(-1, 1) * X + (1 - lam.reshape(-1, 1)) * X_shuffled
        
        # 混合标签（软标签）
        y_mixed = lam * y + (1 - lam) * y_shuffled
        
        return X_mixed, y_mixed
