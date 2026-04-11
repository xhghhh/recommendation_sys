"""
训练监控和可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
import json


class TrainingMonitor:
    """
    训练过程监控器
    用于记录和可视化训练过程中的指标
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化监控器
        
        Args:
            save_dir: 保存目录（可选）
        """
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'train_logloss': [],
            'val_logloss': [],
            'epochs': []
        }
        self.save_dir = save_dir
        self.start_time = None
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def on_train_begin(self):
        """训练开始时的回调"""
        self.start_time = datetime.now()
        print(f"训练开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def on_epoch_end(self, epoch: int,
                    train_loss: float,
                    val_loss: Optional[float] = None,
                    train_metrics: Optional[Dict[str, float]] = None,
                    val_metrics: Optional[Dict[str, float]] = None):
        """
        每个epoch结束时的回调
        
        Args:
            epoch: 当前epoch
            train_loss: 训练损失
            val_loss: 验证损失（可选）
            train_metrics: 训练指标（可选）
            val_metrics: 验证指标（可选）
        """
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        
        if train_metrics:
            if 'auc' in train_metrics:
                self.history['train_auc'].append(train_metrics['auc'])
            if 'logloss' in train_metrics:
                self.history['train_logloss'].append(train_metrics['logloss'])
        
        if val_metrics:
            if 'auc' in val_metrics:
                self.history['val_auc'].append(val_metrics['auc'])
            if 'logloss' in val_metrics:
                self.history['val_logloss'].append(val_metrics['logloss'])
        
        # 打印当前epoch信息
        msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
        if val_metrics and 'auc' in val_metrics:
            msg += f" | Val AUC: {val_metrics['auc']:.4f}"
        
        print(msg)
    
    def on_train_end(self):
        """训练结束时的回调"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        print(f"\n训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总训练时长: {duration}")
        
        # 保存训练历史
        if self.save_dir:
            self.save_history()
    
    def save_history(self):
        """保存训练历史"""
        if not self.save_dir:
            return
        
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"训练历史已保存到: {history_path}")
    
    def plot_losses(self, save_path: Optional[str] = None):
        """
        绘制损失曲线
        
        Args:
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=(10, 6))
        
        epochs = self.history['epochs']
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        
        if self.history['val_loss']:
            plt.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics(self, metric_name: str = 'auc', save_path: Optional[str] = None):
        """
        绘制指标曲线
        
        Args:
            metric_name: 指标名称
            save_path: 保存路径（可选）
        """
        train_key = f'train_{metric_name}'
        val_key = f'val_{metric_name}'
        
        if train_key not in self.history or not self.history[train_key]:
            print(f"没有 {metric_name} 指标的历史记录")
            return
        
        plt.figure(figsize=(10, 6))
        
        epochs = self.history['epochs']
        plt.plot(epochs, self.history[train_key], 'b-', label=f'Train {metric_name.upper()}', linewidth=2)
        
        if val_key in self.history and self.history[val_key]:
            plt.plot(epochs, self.history[val_key], 'r-', label=f'Val {metric_name.upper()}', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.upper())
        plt.title(f'Training and Validation {metric_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.save_dir:
            plt.savefig(os.path.join(self.save_dir, f'{metric_name}_curve.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_all(self):
        """绘制所有监控图表"""
        self.plot_losses()
        
        if self.history['train_auc']:
            self.plot_metrics('auc')
        
        if self.history['train_logloss']:
            self.plot_metrics('logloss')
    
    def get_best_epoch(self, metric: str = 'val_auc', mode: str = 'max') -> tuple:
        """
        获取最佳epoch
        
        Args:
            metric: 指标名称
            mode: 'max' 或 'min'
            
        Returns:
            (最佳epoch, 最佳指标值)
        """
        if metric not in self.history or not self.history[metric]:
            return None, None
        
        values = self.history[metric]
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        best_epoch = self.history['epochs'][best_idx]
        best_value = values[best_idx]
        
        return best_epoch, best_value


class EarlyStopping:
    """
    早停机制
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001,
                 mode: str = 'max', restore_best_weights: bool = True):
        """
        初始化早停器
        
        Args:
            patience: 容忍轮数
            min_delta: 最小改善幅度
            mode: 'max' 或 'min'
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = None
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, value: float, model: Any) -> bool:
        """
        检查是否应该早停
        
        Args:
            value: 当前指标值
            model: 模型对象
            
        Returns:
            是否应该停止训练
        """
        if self.best_value is None:
            self.best_value = value
            self.save_checkpoint(model)
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self.restore_checkpoint(model)
                return True
        
        return False
    
    def save_checkpoint(self, model: Any):
        """保存检查点"""
        import copy
        self.best_weights = copy.deepcopy(model.state_dict() if hasattr(model, 'state_dict') else None)
    
    def restore_checkpoint(self, model: Any):
        """恢复检查点"""
        if self.best_weights is not None and hasattr(model, 'load_state_dict'):
            model.load_state_dict(self.best_weights)
            print("已恢复最佳模型权重")


class ModelCheckpoint:
    """
    模型检查点保存
    """
    
    def __init__(self, save_dir: str, monitor: str = 'val_auc',
                 mode: str = 'max', save_best_only: bool = True):
        """
        初始化检查点保存器
        
        Args:
            save_dir: 保存目录
            monitor: 监控指标
            mode: 'max' 或 'min'
            save_best_only: 是否只保存最佳模型
        """
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = None
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def __call__(self, model: Any, value: float, epoch: int):
        """
        保存检查点
        
        Args:
            model: 模型对象
            value: 当前指标值
            epoch: 当前epoch
        """
        should_save = False
        
        if self.best_value is None:
            should_save = True
            self.best_value = value
        elif self.mode == 'max' and value > self.best_value:
            should_save = True
            self.best_value = value
        elif self.mode == 'min' and value < self.best_value:
            should_save = True
            self.best_value = value
        
        if should_save or not self.save_best_only:
            save_path = os.path.join(self.save_dir, f'model_epoch_{epoch}.pt')
            if hasattr(model, 'save'):
                model.save(save_path)
            print(f"模型已保存到: {save_path}")
