"""
基础模型基类
所有推荐模型的抽象基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime


class BaseModel(ABC):
    """
    推荐系统模型基类
    
    所有具体模型需要继承此类并实现抽象方法
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型
        
        Args:
            config: 模型配置字典
        """
        self.config = config
        self.model_name = config.get('model_name', 'base_model')
        self.model = None
        self.is_trained = False
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': {},
            'val_metrics': {},
            'epochs': []
        }
        
    @abstractmethod
    def build_model(self, **kwargs) -> None:
        """
        构建模型结构
        
        子类需要实现具体的模型构建逻辑
        """
        pass
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            
        Returns:
            训练历史记录
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 输入特征
            
        Returns:
            预测概率数组 (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率分布
        
        Args:
            X: 输入特征
            
        Returns:
            预测概率分布 (n_samples, n_classes)
        """
        pass
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 metrics: Optional[list] = None) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            metrics: 评估指标列表，默认 ['auc', 'logloss', 'accuracy']
            
        Returns:
            评估指标字典
        """
        from src.utils.metrics import calculate_metrics
        
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        y_pred_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)
        
        results = calculate_metrics(y_test, y_pred, y_pred_proba, metrics)
        
        return results
    
    def save(self, save_path: str) -> None:
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 保存配置
        config_path = os.path.join(save_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # 保存训练历史
        history_path = os.path.join(save_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 子类实现具体的模型保存逻辑
        self._save_model_weights(save_path)
        
        print(f"模型已保存到: {save_path}")
    
    @abstractmethod
    def _save_model_weights(self, save_path: str) -> None:
        """
        保存模型权重（子类实现）
        """
        pass
    
    def load(self, load_path: str) -> None:
        """
        加载模型
        
        Args:
            load_path: 加载路径
        """
        # 加载配置
        config_path = os.path.join(load_path, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 加载训练历史
        history_path = os.path.join(load_path, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        # 子类实现具体的模型加载逻辑
        self._load_model_weights(load_path)
        self.is_trained = True
        
        print(f"模型已从 {load_path} 加载")
    
    @abstractmethod
    def _load_model_weights(self, load_path: str) -> None:
        """
        加载模型权重（子类实现）
        """
        pass
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        获取特征重要性（如果模型支持）
        
        Returns:
            特征重要性DataFrame，如果不支持则返回None
        """
        return None
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       n_splits: int = 5,
                       random_state: int = 42) -> Dict[str, list]:
        """
        交叉验证
        
        Args:
            X: 特征数据
            y: 标签数据
            n_splits: 折数
            random_state: 随机种子
            
        Returns:
            每折的评估结果
        """
        from sklearn.model_selection import StratifiedKFold
        from src.utils.metrics import calculate_metrics
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        cv_results = {
            'auc': [],
            'logloss': [],
            'accuracy': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 重新构建和训练模型
            self.build_model()
            self.fit(X_train, y_train, X_val, y_val, verbose=False)
            
            # 评估
            results = self.evaluate(X_val, y_val)
            
            for metric in cv_results.keys():
                cv_results[metric].append(results[metric])
        
        # 计算平均值和标准差
        print("\n" + "="*50)
        print("交叉验证结果:")
        for metric, values in cv_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric}: {mean_val:.4f} (+/- {std_val:.4f})")
        
        return cv_results
