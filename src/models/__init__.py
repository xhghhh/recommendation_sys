"""
模型模块
"""

from src.models.base_model import BaseModel
from src.models.logistic_regression import LogisticRegressionModel

try:
    from src.models.xgboost_model import XGBoostModel
except ImportError:
    XGBoostModel = None

try:
    from src.models.lightgbm_model import LightGBMModel
except ImportError:
    LightGBMModel = None

__all__ = [
    'BaseModel',
    'LogisticRegressionModel',
    'XGBoostModel',
    'LightGBMModel'
]
