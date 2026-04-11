"""
数据增强使用示例
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from src.data.data_loader import DataLoader, FeatureProcessor, split_data
from src.data.data_augmentation import DataAugmentor, SMOTEAugmentor


def main():
    print("="*60)
    print("数据增强示例")
    print("="*60)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    data_loader = DataLoader('demo_1000.parquet')
    df = data_loader.load()
    
    # 获取特征分类
    feature_dict = data_loader.get_feature_columns()
    
    # 2. 特征工程
    print("\n[2/4] 特征工程...")
    processor = FeatureProcessor()
    df_processed = processor.prepare_features_for_ml(df, feature_dict)
    
    # 准备特征和标签
    label_col = 'label_type'
    feature_cols = [col for col in df_processed.columns 
                   if col not in [label_col, 'user_id', 'item_id', 'label_time']]
    
    X = df_processed[feature_cols]
    y = df_processed[label_col] - 1  # 转换为0/1
    
    print(f"特征维度: {len(feature_cols)}")
    print(f"样本数: {len(X)}")
    print(f"标签分布: {y.value_counts().to_dict()}")
    
    # 3. 数据划分
    print("\n[3/4] 划分数据集...")
    train_df, val_df, test_df = split_data(
        pd.concat([X, y], axis=1),
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        stratify_col=label_col
    )
    
    X_train = train_df.drop(columns=[label_col])
    y_train = train_df[label_col]
    
    print(f"训练集大小: {len(X_train)}")
    
    # 4. 数据增强
    print("\n[4/4] 数据增强...")
    
    # 分类特征列
    scalar_cols = [col for col in X_train.columns 
                   if X_train[col].dtype in ['float64', 'int64']]
    
    # 注意：增强前需要区分哪些列是数组/序列（这里简化处理，假设都是标量）
    # 实际使用时需要根据原始特征类型来区分
    
    augmentor = DataAugmentor(random_state=42)
    
    # 方法1: 保守增强
    print("\n方法1: 保守增强 (高斯噪声)")
    X_train_aug1 = augmentor.augment_scalar_gaussian_noise(
        X_train, scalar_cols, sigma_ratio=0.01
    )
    print(f"  增强后训练集大小: {len(X_train_aug1)}")
    
    # 方法2: 激进增强
    print("\n方法2: 激进增强 (噪声+缩放)")
    X_train_aug2 = augmentor.augment_scalar_gaussian_noise(
        X_train, scalar_cols, sigma_ratio=0.05
    )
    X_train_aug2 = augmentor.augment_scalar_random_scale(
        X_train_aug2, scalar_cols, scale_range=(0.9, 1.1)
    )
    print(f"  增强后训练集大小: {len(X_train_aug2)}")
    
    # 方法3: 组合增强（2倍数据）
    print("\n方法3: 组合增强 (2x)")
    # 这里简化处理，只对标量特征增强
    # 实际使用时需要传入array_cols和seq_cols
    X_train_aug3 = pd.concat([X_train, X_train_aug1], ignore_index=True)
    y_train_aug3 = pd.concat([y_train, y_train], ignore_index=True)
    print(f"  增强后训练集大小: {len(X_train_aug3)}")
    
    # 方法4: SMOTE过采样
    print("\n方法4: SMOTE过采样")
    smote_augmentor = SMOTEAugmentor(random_state=42)
    X_train_smote, y_train_smote = smote_augmentor.fit_resample(X_train, y_train)
    
    print("\n" + "="*60)
    print("数据增强示例完成!")
    print("="*60)
    print("\n增强方法对比:")
    print(f"  原始数据: {len(X_train)} 条")
    print(f"  保守增强: {len(X_train_aug1)} 条 (噪声)")
    print(f"  激进增强: {len(X_train_aug2)} 条 (噪声+缩放)")
    print(f"  组合增强: {len(X_train_aug3)} 条 (2x)")
    print(f"  SMOTE: {len(X_train_smote)} 条 (类别平衡)")
    
    print("\n提示: 实际训练时，可以对比不同增强方法的效果")
    print("      推荐使用保守增强或SMOTE作为起点")


if __name__ == '__main__':
    main()
