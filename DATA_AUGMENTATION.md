# 数据增强方案文档

## 1. 背景与动机

### 1.1 当前问题
- **样本量小**: 仅1000条数据
- **过拟合严重**: Baseline模型训练集AUC=1.0，测试集AUC=0.71
- **深度学习受限**: 神经网络参数量大，小数据容易过拟合

### 1.2 目标
通过数据增强，有效扩充训练样本，提升模型泛化能力。

---

## 2. 数据特点分析

### 2.1 特征类型
| 类型 | 列数 | 特点 | 增强策略 |
|------|------|------|----------|
| 标量特征 | 48 | 数值稳定 | 添加高斯噪声 |
| 数组特征 | 22 | 长度可变 | 随机mask、裁剪、拼接 |
| 序列特征 | 45 | 时序相关 | 随机mask、重排序、子序列采样 |
| ID特征 | 2 | 离散值 | 不增强（保持唯一性） |
| 标签 | 1 | 二分类 | 不增强 |

### 2.2 关键观察
- 序列特征长度不均（0~16+）
- 存在大量缺失值（114列）
- 类别不平衡（7:1）

---

## 3. 数据增强策略

### 3.1 标量特征增强

#### 3.1.1 高斯噪声 (Gaussian Noise)
```python
# 对数值特征添加小幅度噪声
noise = np.random.normal(0, sigma, size=feature.shape)
augmented_feature = feature + noise
```
- **适用**: User/Item Int scalar features
- **参数**: sigma = 0.01 * feature_std
- **强度**: 弱

#### 3.1.2 随机缩放 (Random Scaling)
```python
# 随机缩放数值
scale = np.random.uniform(0.95, 1.05)
augmented_feature = feature * scale
```
- **适用**: 所有数值特征
- **强度**: 弱

### 3.2 数组特征增强

#### 3.2.1 随机Mask (Random Masking)
```python
# 随机将数组中的部分元素设为0
mask = np.random.random(array.shape) > mask_ratio
augmented_array = array * mask
```
- **适用**: User Int Array, User Dense
- **参数**: mask_ratio = 0.1 ~ 0.3
- **强度**: 中

#### 3.2.2 随机裁剪 (Random Cropping)
```python
# 随机裁剪数组的一部分
start = random.randint(0, len(array) - crop_length)
augmented_array = array[start:start + crop_length]
```
- **适用**: 长度较长的数组
- **参数**: crop_ratio = 0.8
- **强度**: 中

#### 3.2.3 数组混合 (Array Mixup)
```python
# 两个样本的数组加权混合
alpha = 0.2
augmented_array = alpha * array1 + (1-alpha) * array2
```
- **适用**: User Dense Features
- **强度**: 强

### 3.3 序列特征增强 ⭐核心

#### 3.3.1 序列随机Mask (Sequence Masking)
```python
# 随机mask序列中的部分行为
mask_indices = random.sample(range(seq_length), int(seq_length * mask_ratio))
for idx in mask_indices:
    sequence[idx] = 0  # 或特殊token
```
- **适用**: Domain Sequence Features
- **参数**: mask_ratio = 0.1 ~ 0.2
- **强度**: 中
- **原理**: 模拟用户行为的随机缺失

#### 3.3.2 子序列采样 (Subsequence Sampling)
```python
# 从长序列中采样连续子序列
if len(sequence) > min_length:
    start = random.randint(0, len(sequence) - subseq_length)
    augmented_sequence = sequence[start:start + subseq_length]
```
- **适用**: 长度>5的序列
- **参数**: subseq_length = max_length * 0.8
- **强度**: 中
- **原理**: 用户近期行为比远期更重要

#### 3.3.3 序列重排序 (Sequence Reordering)
```python
# 对序列进行小幅度的重排序
# 保持时序大致不变，局部打乱
chunk_size = 3
chunks = [sequence[i:i+chunk_size] for i in range(0, len(sequence), chunk_size)]
random.shuffle(chunks)
augmented_sequence = [item for chunk in chunks for item in chunk]
```
- **适用**: Domain Sequence
- **强度**: 强
- **注意**: 可能破坏时序信息，谨慎使用

#### 3.3.4 序列拼接 (Sequence Concatenation)
```python
# 将两个用户的序列拼接（需要标签相同）
if label1 == label2:
    augmented_sequence = sequence1 + sequence2[:max_add_length]
```
- **适用**: 短序列样本
- **强度**: 强
- **条件**: 仅同标签样本拼接

### 3.4 样本级别增强

#### 3.4.1 SMOTE (Synthetic Minority Over-sampling)
```python
# 对少数类样本（label=2）进行过采样
# 在特征空间插值生成新样本
```
- **适用**: 解决类别不平衡
- **强度**: 强
- **注意**: 需要在特征工程后的空间进行

#### 3.4.2 样本混合 (Sample Mixup)
```python
# 两个样本按比例混合
alpha = np.random.beta(0.2, 0.2)
augmented_sample = alpha * sample1 + (1-alpha) * sample2
augmented_label = alpha * label1 + (1-alpha) * label2  # 软标签
```
- **适用**: 深度学习训练
- **强度**: 强

---

## 4. 增强策略组合

### 4.1 保守方案 (推荐先尝试)
```
标量: 高斯噪声 (sigma=0.01)
数组: 随机Mask (ratio=0.1)
序列: 随机Mask (ratio=0.1) + 子序列采样
样本: 对少数类SMOTE
增强倍数: 2x
```

### 4.2 激进方案
```
标量: 高斯噪声 + 随机缩放
数组: Mask + 裁剪 + Mixup
序列: Mask + 重排序 + 拼接
样本: Mixup
增强倍数: 5x
```

### 4.3 针对深度学习的方案
```
标量: 高斯噪声 (sigma=0.05)
数组: Mask (ratio=0.2)
序列: Mask (ratio=0.15) + 子序列采样
样本: Mixup (alpha=0.2)
增强倍数: 3x
```

---

## 5. 实现细节

### 5.1 增强流程
```
原始数据 (1000条)
    ↓
特征工程 (207维)
    ↓
数据增强 (3000条)
    ↓
训练模型
```

### 5.2 注意事项

1. **验证集不增强**: 只增强训练集
2. **标签一致性**: 增强后标签保持不变（除Mixup）
3. **ID不增强**: user_id, item_id 保持不变
4. **增强随机性**: 每次epoch使用不同的增强

### 5.3 评估方法

```python
# 对比实验
1. 无增强:  baseline
2. 保守增强: 看是否提升泛化
3. 激进增强: 看是否过拟合减少

评估指标: 测试集AUC、验证集稳定性
```

---

## 6. 预期效果

| 方案 | 训练集大小 | 预期测试AUC | 过拟合程度 |
|------|-----------|------------|-----------|
| 无增强 | 700 | 0.71 | 严重 |
| 保守增强 | 1400 | 0.75 | 中等 |
| 激进增强 | 3500 | 0.78 | 轻微 |

---

## 7. 下一步实现计划

1. **实现基础增强器**: 标量噪声 + 数组Mask + 序列Mask
2. **实现高级增强**: Mixup + SMOTE
3. **集成到训练流程**: 在DataLoader中加入增强
4. **对比实验**: 验证增强效果

---

*文档创建时间: 2024年*
*适用数据集: TAAC2026 Demo Dataset*
