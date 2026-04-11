"""
LightGBM Baseline 模型
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import os

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: lightgbm 未安装，请运行: pip install lightgbm")

from src.models.base_model import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM 模型
    轻量级梯度提升树Baseline
    """
    
    def __init__(self, config: Dict[str, Any]):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm 未安装，请运行: pip install lightgbm")
        
        super().__init__(config)
        self.model_name = 'lightgbm'
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', -1)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.num_leaves = config.get('num_leaves', 31)
        self.subsample = config.get('subsample', 0.8)
        self.colsample_bytree = config.get('colsample_bytree', 0.8)
        self.class_weight = config.get('class_weight', None)
        self.early_stopping_rounds = config.get('early_stopping_rounds', 10)
        
    def build_model(self, **kwargs) -> None:
        """构建LightGBM模型"""
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            class_weight=self.class_weight,
            objective='binary',
            metric='auc',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        print(f"LightGBM模型已构建: n_estimators={self.n_estimators}, num_leaves={self.num_leaves}")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs) -> Dict[str, Any]:
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        print("开始训练LightGBM模型...")
        
        # 准备评估集
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # 训练
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(self.early_stopping_rounds), lgb.log_evaluation(0)] if len(eval_set) > 1 else None
        )
        
        self.is_trained = True
        
        # 记录最佳迭代轮数
        if hasattr(self.model, 'best_iteration_'):
            print(f"最佳迭代轮数: {self.model.best_iteration_}")
            self.training_history['best_iteration'] = self.model.best_iteration_
        
        # 计算指标
        from src.utils.metrics import calculate_metrics
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)
        train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
        
        print(f"训练集指标: AUC={train_metrics['auc']:.4f}, LogLoss={train_metrics['logloss']:.4f}")
        
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print(f"验证集指标: AUC={val_metrics['auc']:.4f}, LogLoss={val_metrics['logloss']:.4f}")
            self.training_history['val_metrics'] = val_metrics
        
        self.training_history['train_metrics'] = train_metrics
        
        return self.training_history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        importance = pd.DataFrame({
            'feature': self.model.feature_name_,
            'importance': self.model.feature_importances_
        })
        
        return importance.sort_values('importance', ascending=False)
    
    def _save_model_weights(self, save_path: str) -> None:
        model_path = os.path.join(save_path, 'model.txt')
        self.model.booster_.save_model(model_path)
    
    def _load_model_weights(self, load_path: str) -> None:
        model_path = os.path.join(load_path, 'model.txt')
        self.model = lgb.Booster(model_file=model_path)
