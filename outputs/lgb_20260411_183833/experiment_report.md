# 实验报告 - LGB Model

**生成时间**: 2026-04-11 18:38:34

---

## 1. 实验概述

### 1.1 模型配置

| 参数 | 值 |
|------|-----|
| model_name | lightgbm |
| n_estimators | 100 |
| num_leaves | 31 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| class_weight | balanced |
| early_stopping_rounds | 10 |

### 1.2 数据信息

| 属性 | 值 |
|------|-----|
| 总样本数 | 1000 |
| 训练集大小 | 700 (70.0%) |
| 验证集大小 | 100 (10.0%) |
| 测试集大小 | 200 (20.0%) |
| 特征维度 | 207 |

### 1.3 标签分布

| 标签 | 数量 | 比例 |
|------|------|------|
| 1 | 876 | 87.60% |
| 2 | 124 | 12.40% |

---

## 2. 实验结果

### 2.1 训练集表现

| 指标 | 值 |
|------|-----|
| AUC | 0.997356 |
| LOGLOSS | 0.478918 |
| ACCURACY | 0.947143 |

### 2.2 验证集表现

| 指标 | 值 |
|------|-----|
| AUC | 0.479167 |
| LOGLOSS | 0.582655 |
| ACCURACY | 0.780000 |

### 2.3 测试集表现

| 指标 | 值 |
|------|-----|
| AUC | 0.648229 |
| LOGLOSS | 0.562022 |
| ACCURACY | 0.805000 |

---

## 3. 可视化结果

### 3.1 ROC 曲线
![ROC Curve](roc_curve.png)

### 3.2 混淆矩阵
![Confusion Matrix](confusion_matrix.png)

---

## 4. 特征重要性

### Top 20 重要特征

| 排名 | 特征名 | 重要性 |
|------|--------|--------|
| 1 | domain_b_seq_72_mean | 7.000000 |
| 2 | item_int_feats_10 | 5.000000 |
| 3 | item_int_feats_5 | 5.000000 |
| 4 | item_int_feats_13 | 4.000000 |
| 5 | domain_c_seq_33_mean | 4.000000 |
| 6 | item_int_feats_16 | 4.000000 |
| 7 | domain_a_seq_45_mean | 4.000000 |
| 8 | domain_c_seq_32_mean | 4.000000 |
| 9 | user_int_feats_15_mean | 3.000000 |
| 10 | domain_c_seq_36_max | 3.000000 |
| 11 | user_dense_feats_61_mean | 3.000000 |
| 12 | domain_d_seq_18_mean | 3.000000 |
| 13 | domain_c_seq_27_len | 3.000000 |
| 14 | domain_a_seq_39_mean | 3.000000 |
| 15 | user_int_feats_63_mean | 3.000000 |
| 16 | domain_c_seq_29_max | 3.000000 |
| 17 | domain_b_seq_79_max | 2.000000 |
| 18 | user_int_feats_48 | 2.000000 |
| 19 | item_int_feats_8 | 2.000000 |
| 20 | domain_d_seq_23_max | 2.000000 |

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
- 测试集 AUC: 0.6482
- 测试集 LogLoss: 0.5620
- 测试集 Accuracy: 0.8050

### 6.2 改进方向
1. **特征工程**: 尝试更多的特征组合和交叉特征
2. **模型调优**: 使用网格搜索或贝叶斯优化调整超参数
3. **集成学习**: 尝试模型融合策略
4. **深度学习**: 实现 DeepFM、DIN 等深度学习模型
5. **序列建模**: 充分利用 45 列 Domain Sequence 特征

---

*本报告由自动化脚本生成*
