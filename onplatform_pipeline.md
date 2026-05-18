
好的，我已经仔细读完了官方 baseline 的全部代码。下面是详细讲解：

---

## 一、代码包含哪些部分

```
baseline_by_organizer/
├── instructions.md    # 平台使用指南（训练/评测流程、环境变量说明）
├── run.sh             # ⭐ 训练入口脚本（平台强制要求）
├── train.py           # 训练主程序（解析参数、加载数据、构建模型、启动训练）
├── dataset.py         # 数据读取（Parquet + schema.json → PyTorch Dataset）
├── model.py           # PCVRHyFormer 模型定义（约1700行，核心模型）
├── trainer.py         # 训练循环 + 验证评测（AUC/LogLoss计算）
├── utils.py           # 工具函数（日志、早停、Focal Loss、随机种子）
└── ns_groups.json     # NS特征分组配置（可选）
```

---

## 二、如何读取数据集

### 数据格式
官方要求的数据目录结构：
```
data_dir/
├── *.parquet          # 训练数据（多文件或单文件）
└── schema.json        # 特征元数据（描述每列的ID、词表大小、维度）
```

### 读取流程（`dataset.py`）

```python
from dataset import get_pcvr_data

train_loader, valid_loader, dataset = get_pcvr_data(
    data_dir=args.data_dir,           # 从环境变量 TRAIN_DATA_PATH 读取
    schema_path=args.schema_path,     # data_dir/schema.json
    batch_size=256,
    valid_ratio=0.1,                  # 最后10% Row Group 做验证
    ...
)
```

**关键设计**：
- 使用 `pyarrow` 直接读取 Parquet，按 **Row Group** 划分训练/验证集
- 不是按样本随机划分，而是按文件块划分（适合大数据集流式读取）
- 每个 batch 是 `Dict[str, torch.Tensor]`，包含所有特征

---

## 三、哪些是输入，哪些是 Target

### 输入特征（`dataset.py` → `_convert_batch`）

从 Parquet 读出的原始列，被整理成以下 **5大类输入**：

| 输入字段 | 类型 | 内容 | 来源 |
|---------|------|------|------|
| `user_int_feats` | `[B, U_int_dim]` | 用户离散特征（标量+数组展平） | `user_int_feats_*` |
| `user_dense_feats` | `[B, U_dense_dim]` | 用户稠密特征（浮点数组） | `user_dense_feats_*` |
| `item_int_feats` | `[B, I_int_dim]` | 物品离散特征 | `item_int_feats_*` |
| `item_dense_feats` | `[B, 0]` | 物品稠密特征（**当前为空**） | - |
| `seq_a / seq_b / seq_c / seq_d` | `[B, n_feats, max_len]` | 4个域的行为序列 | `domain_*_seq_*` |

**序列特征的附加信息**：
- `seq_a_len`, `seq_b_len`, ...：`[B]` 每个样本的实际序列长度
- `seq_a_time_bucket`, ...：`[B, max_len]` 时间分桶编码（用于建模时序）

### Target（标签）

```python
# dataset.py 第512行
labels = (batch.column('label_type') == 2).astype(np.int64)
```

- `label_type == 1` → 标签 `0`（点击/曝光，未转化）
- `label_type == 2` → 标签 `1`（转化，正样本）

所以这是一个 **二分类问题**：预测用户是否会转化。

---

## 四、模型保存到哪里

### 训练阶段

**环境变量**：`TRAIN_CKPT_PATH`（平台自动传入）

**保存代码**（`trainer.py`）：

```python
# 保存目录命名规范：必须以 global_step 开头
dir_name = f"global_step{global_step}.layer=2.head=4.hidden=64.best_model"
ckpt_dir = os.path.join(self.save_dir, dir_name)

# 保存模型权重
torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))

# 同时保存 sidecar 文件（方便评测时复现）
# - schema.json      ← 特征结构
# - ns_groups.json   ← 特征分组（可选）
# - train_config.json ← 训练配置
```

**平台要求**（`instructions.md`）：
1. 模型权重 **必须** 保存到 `TRAIN_CKPT_PATH`
2. 检查点文件夹 **必须以 `global_step` 为前缀**，否则平台无法识别
3. 目录名只能用：字母、数字、`_`、`-`、`=`、`.`

---

## 五、如何评测

### 训练时验证（`trainer.py`）

每个 epoch 结束时自动验证：

```python
def evaluate(self):
    # 1. 前向传播得到 logits
    logits = model(model_input)  # [B, 1]
    
    # 2. 计算 AUC（sklearn）
    probs = torch.sigmoid(logits).numpy()
    auc = roc_auc_score(labels, probs)
    
    # 3. 计算 LogLoss
    logloss = F.binary_cross_entropy_with_logits(logits, labels)
    
    return auc, logloss
```

### 平台评测（提交后）

评测分为两个阶段：

#### 1. 推理阶段（你需要提交 `infer.py`）

虽然 baseline 目录里没有 `infer.py`，但根据 `instructions.md`，评测时必须提交：

- **文件名必须严格是 `infer.py`**
- 必须包含 **无参 `main()` 函数**
- 输出路径：`EVAL_RESULT_PATH/predictions.json`

```python
# infer.py 的核心逻辑（你需要自己写或基于baseline改造）
def main():
    # 1. 加载模型（从 MODEL_OUTPUT_PATH）
    # 2. 读取测试数据（EVAL_DATA_PATH）
    # 3. 前向推理
    # 4. 输出 predictions.json
```

#### 2. predictions.json 格式

```json
{
    "predictions": {
        "3864676": 0.8732,   // user_id → 预测转化率（0~1之间）
        "8507274": 0.1245,
        ...
    }
}
```

**关键约束**：
- 必须严格匹配测试集所有合法 `user_id`
- 不能缺失、不能多出多余 ID
- 每个队伍24小时内最多提交3次

---

## 六、整体流程总结

```
训练阶段：
├─ run.sh 被平台自动执行
│   └─ python train.py [参数]
│       ├─ 读取 TRAIN_DATA_PATH/*.parquet + schema.json
│       ├─ 构建 PCVRHyFormer 模型
│       ├─ 训练（BCE/Focal Loss）
│       ├─ 验证（AUC + LogLoss）
│       └─ 保存最佳模型到 TRAIN_CKPT_PATH/global_step*.best_model/model.pt

评测阶段：
├─ 在平台选择已发布的模型
├─ 上传 infer.py（+ prepare.sh 安装依赖）
├─ 平台自动执行：
│   └─ infer.py 的 main() 函数
│       ├─ 读取 EVAL_DATA_PATH 测试数据
│       ├─ 加载 MODEL_OUTPUT_PATH 的模型
│       ├─ 推理生成 predictions.json
│       └─ 保存到 EVAL_RESULT_PATH/predictions.json
└─ 平台计算最终分数
```

---

## 七、这个 baseline 模型的核心特点

`PCVRHyFormer`（`model.py`）是一个 **Transformer-based** 的深度学习模型：

| 组件 | 说明 |
|------|------|
| **Embedding 层** | 离散特征 → 稠密向量（支持超大词表的 skip 策略） |
| **NS Tokenizer** | 将 user/item 特征分组压缩成 token |
| **序列编码器** | Transformer/SwiGLU/Longer 三种可选，处理 4 个 domain 的序列 |
| **Attention** | 多头自注意力 + RoPE 位置编码（可选） |
| **RankMixer** | 特征交互模块 |
| **分类头** | 输出转化概率 |

---

需要我进一步讲解：
1. **如何基于这个 baseline 改造/优化**？
2. **infer.py 的具体实现**？
3. **模型的 Attention 机制细节**？