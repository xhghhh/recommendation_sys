"""
XGBoost Baseline 模型
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import pickle
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: xgboost 未安装，请运行: pip install xgboost")

from src.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost 模型
    梯度提升树Baseline
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化XGBoost模型
        
        Args:
            config: 配置字典
                - n_estimators: 树的数量（默认100）
                - max_depth: 树的最大深度（默认6）
                - learning_rate: 学习率（默认0.1）
                - subsample: 子采样比例（默认0.8）
                - colsample_bytree: 列采样比例（默认0.8）
                - scale_pos_weight: 正负样本权重比（默认自动计算）
                - early_stopping_rounds: 早停轮数（默认10）
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost 未安装，请运行: pip install xgboost")
        
        super().__init__(config)
        self.model_name = 'xgboost'
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 6)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.subsample = config.get('subsample', 0.8)
        self.colsample_bytree = config.get('colsample_bytree', 0.8)
        self.scale_pos_weight = config.get('scale_pos_weight', None)
        self.early_stopping_rounds = config.get('early_stopping_rounds', 10)
        
    def build_model(self, **kwargs) -> None:
        """构建XGBoost模型"""
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        print(f"XGBoost模型已构建: n_estimators={self.n_estimators}, max_depth={self.max_depth}")
    
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
            训练历史
        """
        if self.model is None:
            self.build_model()
        
        print("开始训练XGBoost模型...")
        
        # 准备评估集
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # 训练 - XGBoost 3.x 简化版
        print(f"Training with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # 记录最佳迭代轮数
        if hasattr(self.model, 'best_iteration'):
            print(f"最佳迭代轮数: {self.model.best_iteration}")
            self.training_history['best_iteration'] = self.model.best_iteration
        
        # 计算训练集指标
        from src.utils.metrics import calculate_metrics
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)
        train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
        
        print(f"训练集指标: AUC={train_metrics['auc']:.4f}, LogLoss={train_metrics['logloss']:.4f}")
        
        # 验证集指标
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print(f"验证集指标: AUC={val_metrics['auc']:.4f}, LogLoss={val_metrics['logloss']:.4f}")
        else:
            val_metrics = None
        
        # 记录训练历史
        self.training_history['train_metrics'] = train_metrics
        if val_metrics:
            self.training_history['val_metrics'] = val_metrics
        
        return self.training_history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 输入特征
            
        Returns:
            预测类别
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 输入特征
            
        Returns:
            预测概率 (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            特征重要性DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        importance = pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        })
        
        return importance.sort_values('importance', ascending=False)
    
    def _save_model_weights(self, save_path: str) -> None:
        """保存模型权重"""
        model_path = os.path.join(save_path, 'model.json')
        self.model.save_model(model_path)
    
    def _load_model_weights(self, load_path: str) -> None:
        """加载模型权重"""
        model_path = os.path.join(load_path, 'model.json')
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
