"""
评估指标模块
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Optional, List


def calculate_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_pred_proba: np.ndarray,
                     metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率
        metrics: 需要计算的指标列表，默认 ['auc', 'logloss', 'accuracy']
        
    Returns:
        指标字典
    """
    if metrics is None:
        metrics = ['auc', 'logloss', 'accuracy']
    
    results = {}
    
    # 二分类问题，取正类的概率
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
        y_pred_proba_pos = y_pred_proba[:, 1]
    else:
        y_pred_proba_pos = y_pred_proba
    
    for metric in metrics:
        metric = metric.lower()
        
        try:
            if metric == 'auc':
                results['auc'] = roc_auc_score(y_true, y_pred_proba_pos)
                
            elif metric == 'logloss':
                # 确保概率在有效范围内
                y_pred_proba_pos_clip = np.clip(y_pred_proba_pos, 1e-15, 1 - 1e-15)
                results['logloss'] = log_loss(y_true, y_pred_proba_pos_clip)
                
            elif metric == 'accuracy':
                results['accuracy'] = accuracy_score(y_true, y_pred)
                
            elif metric == 'precision':
                results['precision'] = precision_score(y_true, y_pred, average='binary')
                
            elif metric == 'recall':
                results['recall'] = recall_score(y_true, y_pred, average='binary')
                
            elif metric == 'f1':
                results['f1'] = f1_score(y_true, y_pred, average='binary')
                
        except Exception as e:
            print(f"计算指标 {metric} 时出错: {e}")
            results[metric] = np.nan
    
    return results


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    打印分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    """
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['Class 1', 'Class 2']))


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         save_path: Optional[str] = None) -> None:
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 1', 'Predicted 2'],
                yticklabels=['Actual 1', 'Actual 2'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                   save_path: Optional[str] = None) -> float:
    """
    绘制ROC曲线
    
    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        save_path: 保存路径（可选）
        
    Returns:
        AUC值
    """
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    
    # 二分类问题，取正类的概率
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
        y_pred_proba_pos = y_pred_proba[:, 1]
    else:
        y_pred_proba_pos = y_pred_proba
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba_pos)
    auc = roc_auc_score(y_true, y_pred_proba_pos)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存到: {save_path}")
    
    plt.show()
    
    return auc


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                save_path: Optional[str] = None) -> None:
    """
    绘制Precision-Recall曲线
    
    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        save_path: 保存路径（可选）
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt
    
    # 二分类问题，取正类的概率
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
        y_pred_proba_pos = y_pred_proba[:, 1]
    else:
        y_pred_proba_pos = y_pred_proba
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba_pos)
    ap = average_precision_score(y_true, y_pred_proba_pos)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR曲线已保存到: {save_path}")
    
    plt.show()


def get_threshold_by_f1(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple:
    """
    根据F1分数找到最佳阈值
    
    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        
    Returns:
        (最佳阈值, 最佳F1分数)
    """
    from sklearn.metrics import f1_score
    
    # 二分类问题，取正类的概率
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
        y_pred_proba_pos = y_pred_proba[:, 1]
    else:
        y_pred_proba_pos = y_pred_proba
    
    thresholds = np.arange(0.1, 1.0, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba_pos >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1
