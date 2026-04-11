# recommendation_sys

TAAC2026 推荐系统竞赛 - Demo 数据集分析与建模

## 任务定义

### 目标
预测用户与物品的交互类型（CTR/CVR 二分类任务）

- **输入**: 用户特征、物品特征、跨域行为序列
- **输出**: `label_type` 概率值
  - `1`: 用户点击/曝光（正样本，占比 87.6%）
  - `2`: 用户转化/购买（负样本，占比 12.4%）

### 评估指标
- **AUC（Area Under ROC Curve）**
  - ROC曲线下面积，衡量二分类模型排序能力的指标
  - 取值范围 [0, 1]，值越大模型性能越好
  - 物理含义：随机抽取一个正样本和一个负样本，模型将正样本预测为正的概率大于负样本预测为正的概率的概率
  - AUC = 0.5 表示随机猜测，AUC = 1.0 表示完美分类
  - 对样本分布不敏感，适合评估不平衡数据集
- **LogLoss（对数损失）**
  - 衡量预测概率与真实标签的差异
  - 值越小越好，惩罚错误预测（尤其是自信的错误预测）

---

## 数据集介绍

### 基本信息
| 属性 | 值 |
|------|-----|
| 文件 | `demo_1000.parquet` |
| 样本数 | 1,000 |
| 特征列 | 120 |
| 唯一用户数 | 1,000 |
| 唯一物品数 | 837 |
| 时间窗口 | ~883秒 |

### 特征分类（120列）

| 类别 | 列数 | 说明 |
|------|------|------|
| ID & Label | 5 | user_id, item_id, label_type, label_time, timestamp |
| User Int Features | 46 | 35列标量 + 11列数组 |
| User Dense Features | 10 | 浮点数组特征 |
| Item Int Features | 14 | 13列标量 + 1列数组 |
| Domain Sequence Features | 45 | 跨域行为序列（A/B/C/D四个域）|

### 数据特点
- 每个用户仅有一条记录（冷启动场景）
- 114列存在缺失值（主要是数组类型）
- 正负样本比例约 7:1，存在不平衡
- 包含丰富的跨域序列特征

---

## 环境依赖

### 虚拟环境
```bash
# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 核心依赖包
```
pandas==2.2.3
pyarrow==19.0.1
scikit-learn
xgboost
lightgbm
numpy
matplotlib
seaborn
jupyter
ipykernel
```

### 安装命令
```bash
pip install pandas==2.2.3 pyarrow==19.0.1 scikit-learn xgboost lightgbm numpy matplotlib seaborn jupyter imbalanced-learn ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Jupyter Kernel 配置
```bash
python -m ipykernel install --user --name=rec_sys --display-name="Python (rec_sys)"
```

然后在 Jupyter 中选择 **"Python (rec_sys)"** 内核。

---

## 文件结构

```
recommendation_sys/
├── README.md                      # 项目说明
├── demo_1000.parquet             # 数据集（1000条样本）
├── demo_1000_analysis.ipynb      # 数据分析 Notebook
├── demo_1000_analysis.html       # 分析结果 HTML
├── train_baseline.py             # Baseline 训练脚本
├── src/                          # 源代码目录
│   ├── models/                   # 模型实现
│   │   ├── base_model.py         # 模型基类
│   │   ├── logistic_regression.py
│   │   ├── xgboost_model.py
│   │   └── lightgbm_model.py
│   ├── data/                     # 数据处理
│   │   └── data_loader.py
│   └── utils/                    # 工具函数
│       ├── metrics.py            # 评估指标
│       └── trainer.py            # 训练监控
├── outputs/                      # 训练输出（自动创建）
└── venv/                         # 虚拟环境
```

---

## 快速开始

### 方式1：运行 Baseline 训练

1. **激活环境**
   ```bash
   source venv/bin/activate
   ```

2. **训练模型**
   ```bash
   # 逻辑回归
   python train_baseline.py --model lr
   
   # XGBoost
   python train_baseline.py --model xgb
   
   # LightGBM
   python train_baseline.py --model lgb
   ```

3. **查看结果**
   - 训练结果保存在 `outputs/{model}_{timestamp}/` 目录
   - 包含：模型文件、评估指标、ROC曲线、混淆矩阵、特征重要性

### 方式2：Jupyter Notebook 分析

1. **激活环境**
   ```bash
   source venv/bin/activate
   ```

2. **启动 Jupyter**
   ```bash
   jupyter notebook
   ```

3. **打开 Notebook**
   - 选择 `demo_1000_analysis.ipynb`
   - 确认内核为 `Python (rec_sys)`
   - 运行所有单元格

---

## 建模思路

### 特征处理
- **标量特征**: 直接输入或Embedding
- **数组特征**: Mean/Max池化
- **序列特征**: Attention/DIN/Transformer（核心）

### 模型架构建议
```
输入层
  ├── ID Embedding (user_id, item_id)
  ├── 标量特征拼接
  ├── 数组特征池化
  └── 序列特征Attention
        ↓
特征融合层 (Concatenate)
        ↓
MLP (多层感知机)
        ↓
Sigmoid输出 (预测概率)
```

---

## 注意事项

1. **版本兼容性**: pyarrow 19.0.1 与 pandas 2.2.3 已验证兼容，请勿随意升级
2. **内存占用**: 序列特征较多，注意批处理大小
3. **数据划分**: 每个用户一条记录，需按用户划分训练/测试集
4. **缺失值**: 数组类型特征存在缺失，需做填充或掩码处理

---

## 参考

- [TAAC2026 竞赛官网](https://algo.qq.com/)
- 数据集 Schema 详见 `demo_1000_analysis.ipynb`
