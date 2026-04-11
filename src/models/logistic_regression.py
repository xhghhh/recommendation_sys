"""
逻辑回归 Baseline 模型
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import os

from src.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    逻辑回归模型
    作为最基础的Baseline
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化逻辑回归模型
        
        Args:
            config: 配置字典
                - C: 正则化强度（默认1.0）
                - penalty: 正则化类型（默认'l2'）
                - max_iter: 最大迭代次数（默认1000）
                - class_weight: 类别权重（默认None，可设为'balanced'）
                - feature_selection: 是否进行特征选择（默认False）
                - k_best: 选择top k特征（默认50）
        """
        super().__init__(config)
        self.model_name = 'logistic_regression'
        self.C = config.get('C', 1.0)
        self.penalty = config.get('penalty', 'l2')
        self.max_iter = config.get('max_iter', 1000)
        self.class_weight = config.get('class_weight', None)
        self.feature_selection = config.get('feature_selection', False)
        self.k_best = config.get('k_best', 50)
        
        self.feature_selector = None
        self.selected_features = None
        
    def build_model(self, **kwargs) -> None:
        """构建逻辑回归模型"""
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver='lbfgs' if self.penalty == 'l2' else 'liblinear',
            random_state=42
        )
        print(f"逻辑回归模型已构建: C={self.C}, penalty={self.penalty}")
    
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
        
        # 特征选择
        if self.feature_selection and X_train.shape[1] > self.k_best:
            print(f"进行特征选择，选择 top {self.k_best} 个特征...")
            self.feature_selector = SelectKBest(score_func=f_classif, k=self.k_best)
            X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            self.selected_features = X_train.columns[self.feature_selector.get_support()].tolist()
            print(f"选择的特征: {self.selected_features[:10]}...")
        else:
            X_train_selected = X_train
            self.selected_features = X_train.columns.tolist()
        
        print("开始训练逻辑回归模型...")
        self.model.fit(X_train_selected, y_train)
        self.is_trained = True
        
        # 计算训练集指标
        from src.utils.metrics import calculate_metrics
        y_train_pred = self.model.predict(X_train_selected)
        y_train_proba = self.model.predict_proba(X_train_selected)
        train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
        
        print(f"训练集指标: AUC={train_metrics['auc']:.4f}, LogLoss={train_metrics['logloss']:.4f}")
        
        # 验证集指标
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print(f"验证集指标: AUC={val_metrics['auc']:.4f}, LogLoss={val_metrics['logloss']:.4f}")
        
        # 记录训练历史
        self.training_history['train_metrics'] = train_metrics
        if X_val is not None:
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
        
        X_selected = self._select_features(X)
        return self.model.predict(X_selected)
    
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
        
        X_selected = self._select_features(X)
        return self.model.predict_proba(X_selected)
    
    def _select_features(self, X: pd.DataFrame) -> np.ndarray:
        """选择特征"""
        if self.feature_selector is not None:
            return self.feature_selector.transform(X)
        return X
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性（系数）
        
        Returns:
            特征重要性DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        importance = pd.DataFrame({
            'feature': self.selected_features,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        })
        
        return importance.sort_values('abs_coefficient', ascending=False)
    
    def _save_model_weights(self, save_path: str) -> None:
        """保存模型权重"""
        model_path = os.path.join(save_path, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_selector': self.feature_selector,
                'selected_features': self.selected_features
            }, f)
    
    def _load_model_weights(self, load_path: str) -> None:
        """加载模型权重"""
        model_path = os.path.join(load_path, 'model.pkl')
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_selector = data['feature_selector']
            self.selected_features = data['selected_features']
