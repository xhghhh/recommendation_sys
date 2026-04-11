"""
Baseline 模型训练脚本
"""

import os
import sys
import json
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_loader import DataLoader, FeatureProcessor, split_data
from src.models.logistic_regression import LogisticRegressionModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.utils.metrics import calculate_metrics, plot_roc_curve, plot_confusion_matrix
from src.utils.trainer import TrainingMonitor


def generate_experiment_report(output_dir: str, model_name: str, config: dict,
                               data_info: dict, metrics: dict,
                               feature_importance: pd.DataFrame = None):
    """
    生成 Markdown 格式的实验报告
    
    Args:
        output_dir: 输出目录
        model_name: 模型名称
        config: 模型配置
        data_info: 数据信息
        metrics: 评估指标
        feature_importance: 特征重要性
    """
    from datetime import datetime
    
    report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    md_content = f"""# 实验报告 - {model_name.upper()} Model

**生成时间**: {report_time}

---

## 1. 实验概述

### 1.1 模型配置

| 参数 | 值 |
|------|-----|
"""
    
    # 添加配置参数
    for key, value in config.items():
        md_content += f"| {key} | {value} |\n"
    
    md_content += f"""
### 1.2 数据信息

| 属性 | 值 |
|------|-----|
| 总样本数 | {data_info['total_samples']} |
| 训练集大小 | {data_info['train_size']} ({data_info['train_size']/data_info['total_samples']*100:.1f}%) |
| 验证集大小 | {data_info['val_size']} ({data_info['val_size']/data_info['total_samples']*100:.1f}%) |
| 测试集大小 | {data_info['test_size']} ({data_info['test_size']/data_info['total_samples']*100:.1f}%) |
| 特征维度 | {data_info['feature_count']} |

### 1.3 标签分布

| 标签 | 数量 | 比例 |
|------|------|------|
"""
    
    # 添加标签分布
    for label, count in data_info['label_distribution'].items():
        ratio = count / data_info['total_samples'] * 100
        md_content += f"| {label} | {count} | {ratio:.2f}% |\n"
    
    md_content += """
---

## 2. 实验结果

### 2.1 训练集表现

| 指标 | 值 |
|------|-----|
"""
    
    # 添加训练集指标
    for metric, value in metrics['train'].items():
        md_content += f"| {metric.upper()} | {value:.6f} |\n"
    
    md_content += """
### 2.2 验证集表现

| 指标 | 值 |
|------|-----|
"""
    
    # 添加验证集指标
    for metric, value in metrics['val'].items():
        md_content += f"| {metric.upper()} | {value:.6f} |\n"
    
    md_content += """
### 2.3 测试集表现

| 指标 | 值 |
|------|-----|
"""
    
    # 添加测试集指标
    for metric, value in metrics['test'].items():
        md_content += f"| {metric.upper()} | {value:.6f} |\n"
    
    md_content += """
---

## 3. 可视化结果

### 3.1 ROC 曲线
![ROC Curve](roc_curve.png)

### 3.2 混淆矩阵
![Confusion Matrix](confusion_matrix.png)

"""
    
    # 添加特征重要性
    if feature_importance is not None and not feature_importance.empty:
        md_content += """---

## 4. 特征重要性

### Top 20 重要特征

| 排名 | 特征名 | 重要性 |
|------|--------|--------|
"""
        
        top_features = feature_importance.head(20)
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            # 根据特征重要性类型选择正确的列名
            if 'importance' in row:
                importance_val = row['importance']
            elif 'coefficient' in row:
                importance_val = abs(row['coefficient'])
            elif 'abs_coefficient' in row:
                importance_val = row['abs_coefficient']
            else:
                importance_val = 0
            
            feature_name = row.get('feature', 'Unknown')
            md_content += f"| {idx} | {feature_name} | {importance_val:.6f} |\n"
    
    md_content += f"""
---

## 5. 文件说明

| 文件名 | 说明 |
|--------|------|
| `results.json` | 实验结果（JSON格式） |
| `roc_curve.png` | ROC曲线可视化 |
| `confusion_matrix.png` | 混淆矩阵可视化 |
| `feature_importance.csv` | 特征重要性详细数据 |
| `model/` | 保存的模型文件 |
| `experiment_report.md` | 本实验报告 |

---

## 6. 结论与建议

### 6.1 主要发现
- 测试集 AUC: {metrics['test'].get('auc', 'N/A'):.4f}
- 测试集 LogLoss: {metrics['test'].get('logloss', 'N/A'):.4f}
- 测试集 Accuracy: {metrics['test'].get('accuracy', 'N/A'):.4f}

### 6.2 改进方向
1. **特征工程**: 尝试更多的特征组合和交叉特征
2. **模型调优**: 使用网格搜索或贝叶斯优化调整超参数
3. **集成学习**: 尝试模型融合策略
4. **深度学习**: 实现 DeepFM、DIN 等深度学习模型
5. **序列建模**: 充分利用 45 列 Domain Sequence 特征

---

*本报告由自动化脚本生成*
"""
    
    # 保存 Markdown 报告
    report_path = os.path.join(output_dir, 'experiment_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"实验报告已保存到: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train Baseline Models')
    parser.add_argument('--data_path', type=str, default='demo_1000.parquet',
                       help='数据文件路径')
    parser.add_argument('--model', type=str, default='lr', choices=['lr', 'xgb', 'lgb'],
                       help='模型类型: lr(逻辑回归), xgb(XGBoost), lgb(LightGBM)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.model}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("TAAC2026 Demo Dataset - Baseline Training")
    print("="*60)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    data_loader = DataLoader(args.data_path)
    df = data_loader.load()
    
    # 获取特征分类
    feature_dict = data_loader.get_feature_columns()
    print(f"\n特征分类:")
    for key, cols in feature_dict.items():
        print(f"  {key}: {len(cols)} 列")
    
    # 2. 数据划分
    print(f"\n[2/5] 划分数据集 (test={args.test_size}, val={args.val_size})...")
    train_df, val_df, test_df = split_data(
        df, 
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=42,
        stratify_col='label_type'
    )
    
    # 3. 特征工程
    print("\n[3/5] 特征工程...")
    processor = FeatureProcessor()
    
    # 处理训练集
    print("  处理训练集...")
    train_processed = processor.prepare_features_for_ml(train_df, feature_dict)
    
    # 处理验证集和测试集（使用训练集的处理器）
    print("  处理验证集...")
    val_processed = processor.prepare_features_for_ml(val_df, feature_dict)
    
    print("  处理测试集...")
    test_processed = processor.prepare_features_for_ml(test_df, feature_dict)
    
    # 准备特征和标签
    label_col = 'label_type'
    feature_cols = [col for col in train_processed.columns 
                   if col not in [label_col, 'user_id', 'item_id', 'label_time']]
    
    X_train = train_processed[feature_cols]
    y_train = train_processed[label_col] - 1  # 转换为0/1
    
    X_val = val_processed[feature_cols]
    y_val = val_processed[label_col] - 1
    
    X_test = test_processed[feature_cols]
    y_test = test_processed[label_col] - 1
    
    print(f"\n特征维度: {len(feature_cols)}")
    print(f"训练集: {X_train.shape}, 标签分布: {y_train.value_counts().to_dict()}")
    
    # 4. 训练模型
    print(f"\n[4/5] 训练 {args.model.upper()} 模型...")
    
    if args.model == 'lr':
        config = {
            'model_name': 'logistic_regression',
            'C': 1.0,
            'penalty': 'l2',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'feature_selection': True,
            'k_best': 50
        }
        model = LogisticRegressionModel(config)
    elif args.model == 'xgb':
        config = {
            'model_name': 'xgboost',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 10
        }
        model = XGBoostModel(config)
    elif args.model == 'lgb':
        config = {
            'model_name': 'lightgbm',
            'n_estimators': 100,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'class_weight': 'balanced',
            'early_stopping_rounds': 10
        }
        model = LightGBMModel(config)
    else:
        raise NotImplementedError(f"模型 {args.model} 尚未实现")
    
    # 训练
    model.fit(X_train, y_train, X_val, y_val)
    
    # 5. 评估
    print("\n[5/5] 模型评估...")
    
    # 训练集评估
    print("\n训练集表现:")
    train_metrics = model.evaluate(X_train, y_train)
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 验证集评估
    print("\n验证集表现:")
    val_metrics = model.evaluate(X_val, y_val)
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 测试集评估
    print("\n测试集表现:")
    test_metrics = model.evaluate(X_test, y_test)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 保存结果
    results = {
        'config': config,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_count': len(feature_cols),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    }
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {results_path}")
    
    # 保存模型
    model.save(os.path.join(output_dir, 'model'))
    
    # 获取特征重要性
    importance = model.get_feature_importance()
    
    # 生成 Markdown 实验报告
    generate_experiment_report(
        output_dir=output_dir,
        model_name=args.model,
        config=config,
        data_info={
            'total_samples': len(df),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'feature_count': len(feature_cols),
            'label_distribution': df['label_type'].value_counts().to_dict()
        },
        metrics={
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        },
        feature_importance=importance
    )
    
    # 可视化
    print("\n生成可视化...")
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    # ROC曲线
    plot_roc_curve(y_test, y_test_proba, 
                   save_path=os.path.join(output_dir, 'roc_curve.png'))
    
    # 混淆矩阵
    plot_confusion_matrix(y_test, y_test_pred,
                         save_path=os.path.join(output_dir, 'confusion_matrix.png'))
    
    # 特征重要性
    if importance is not None:
        importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        print(f"\nTop 10 重要特征:")
        print(importance.head(10))
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"输出目录: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
