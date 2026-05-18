# Tencent Angel Machine Learning Platform User Guide
## 腾讯Angel机器学习平台 用户指南 全文完整翻译

### 模型训练
#### 创建并提交训练任务
1. 填写**任务名称(Job Name)**
2. 填写**任务描述(Job Description)**
3. 点击「本地上传(Local Upload)」从本地上传脚本文件；或点击「新建脚本(New Script)」在线编写新脚本。

**重要注意**：训练模板中**必须包含 run.sh 文件**作为执行入口，任务启动时系统会自动运行该文件！

4. 确认配置后，点击「提交(Submit)」保存任务。

#### 运行任务
点击「运行(Run)」即可启动任务。

#### 查看实例状态
点击运行后，实例状态会自动变为：**训练任务运行中(Training Job Running)**。

#### 查看输出与日志
选中目标任务，点击「实例(Instances)」，进入页面可查看**输出(Output)**和**日志(Logs)**，均可点击查看详情；也可点击「停止(Stop)」终止正在运行的任务。

#### 复制/删除/编辑任务
点击「更多(More)」，可对当前任务进行复制、删除、编辑操作。

---

### 模型发布与管理
#### 模型发布步骤
1. 从训练任务页面点击「实例(Instances)」
2. 点击「输出(Output)」
3. 选中需要发布的检查点(ckpt)，点击「发布(Publish)」，填写**模型名称(Model Name)**和**模型描述(Model Description)**，确认后点击提交。
4. 提交成功后，发布状态会自动变为**已发布(Released)**。
5. 点击左侧菜单栏「模型管理(Model Management)」，即可查看和管理所有已发布模型。

#### 模型编辑与删除
可在模型管理页面对已发布模型进行编辑、删除操作。

---

### 模型评测
#### 评测入口
方式一：在「模型管理」页面，找到待评测模型，点击模型卡片底部的**模型评测**按钮。
方式二：进入「模型评测」页面，点击「创建评测(Create Evaluation)」。

#### 提交评测任务
##### 操作步骤
1. **选择模型**：从下拉列表中选中你要评测的已训练模型。
2. **上传推理代码**：通过「本地上传」上传推理脚本，或「新建脚本」在线编写。

- 必须包含固定命名的入口脚本：**infer.py**
- 该文件内必须定义**无参的 main() 函数**
- 可额外上传其他依赖脚本（如dataset.py、model.py及其他自定义模块），所有上传文件会统一放在 EVAL_INFER_PATH 目录下
- 所有上传脚本总大小限制：**100MB**
3. 点击「提交(Submit)」创建评测任务。

##### 重要注意事项
1. 推理脚本**文件名必须严格为 infer.py**，且必须包含无参 main() 函数，否则评测直接失败。
2. 每日提交次数每24小时在**AOE时间 15:59:59**重置；每支队伍滚动24小时内最多可提交**3次**评测任务；失败、手动终止的任务**不占用**每日次数名额。
3. 无论工业赛道还是学术赛道，单条推理任务**最大限时30分钟**，超时系统将自动终止任务。

##### 评测任务状态说明
| 状态 | 说明 |
| ---- | ---- |
| Pending | 已提交，任务排队中 |
| Waiting for Inference Resources | 等待推理计算资源分配 |
| Inference Running | infer.py 脚本正在执行 |
| Waiting for Evaluation Resources | 推理完成，等待评分资源 |
| Evaluation Running | 平台正在对你的预测结果进行打分 |
| Success | 评测完成，可查看最终分数 |
| Failed | 评测失败，可查看日志排查原因 |

---

### 安装依赖包
若需要执行shell命令安装依赖（类似训练阶段的run.sh），创建评测任务时可上传固定命名脚本：**prepare.sh**。
该脚本会在推理启动前**自动执行**，且运行在已激活的Conda环境中。

脚本模板说明：
```bash
#!/bin/bash
# 1. 该脚本会在推理逻辑启动前自动运行
# 2. 脚本运行于已激活的conda环境内部
# 3. 可在此执行自定义环境配置命令，例如：
# conda install 包名
# 或其他所需环境部署命令
```

---

### 平台环境配置
训练和评测共用统一的软硬件环境。

#### 硬件环境
平台部署高性能GPU并做虚拟化资源分配，单GPU分区配置：
| 资源项 | 配置规格 |
| ---- | ---- |
| 算力 | 单张GPU 20%算力 |
| 显存 | 19GiB |
| CPU核心 | 9核 |
| 内存 | 55GiB |

#### 软件环境
操作系统：Ubuntu 22.04
预装基础软件版本：
- cuda：12.6
- cudnn：9.5.1
- cublas：12.6.3.3
- nccl：2.26.2+cuda12.6
- conda：26.1.1
- python：3.10.20

同时预装大量Python第三方库（文档附完整版本列表）；
如需安装未预装的包，可直接使用 `conda` 或 `pip3` 命令安装。
示例：
```bash
# conda安装
conda install -y pandas
# pip安装
pip3 install pandas
```

---

### 模型训练 环境变量 & 输出规范
#### 1. 训练环境变量
平台提供训练所需目录（训练数据、输出、日志等），以环境变量形式传入容器，可在脚本中读取使用：

| 环境变量名 | 说明 |
| ---- | ---- |
| USER_CACHE_PATH | 用户缓存路径，配额20GB；训练、评测阶段共享，可跨阶段传文件 |
| TRAIN_DATA_PATH | 训练数据集路径 |
| TRAIN_CKPT_PATH | 模型检查点保存路径 |
| TRAIN_TF_EVENTS_PATH | TensorBoard日志文件路径 |

Shell脚本读取方式：`${TRAIN_DATA_PATH}`
Python脚本读取方式：
```python
import os
os.environ.get("TRAIN_DATA_PATH")
```

#### 2. 训练输出规范
##### 2.1 模型权重输出要求
1. 模型权重**必须保存到 TRAIN_CKPT_PATH 指定目录**
2. 迭代检查点文件夹**必须以 global_step 为前缀**，否则平台无法识别；目录名称字符长度不超过300位
3. 目录名仅允许使用：大小写字母、数字、下划线(_)、连字符(-)、等号(=)、点(.)
4. 可在global_step后追加参数信息，示例：
`global_step20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200`

##### 2.2 TensorBoard指标
PyTorch使用参考官方文档；
通过读取 `TRAIN_TF_EVENTS_PATH` 初始化日志写入器，**平台仅支持标量指标**。
示例代码：
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
```

---

### 模型评测 环境变量 & 输出规范
#### 1. 评测环境变量
评测阶段专用目录以环境变量传入：

| 环境变量名 | 说明 |
| ---- | ---- |
| USER_CACHE_PATH | 20GB用户缓存，训练评测共享 |
| MODEL_OUTPUT_PATH | 模型输出路径 |
| EVAL_DATA_PATH | 测试推理数据集路径 |
| EVAL_RESULT_PATH | 中间结果与预测文件输出路径 |
| EVAL_INFER_PATH | 上传的推理脚本所在目录 |

读取方式同训练阶段，Shell直接引用变量，Python用os.environ读取。

#### 2. 评测输出强制规范
1. 必须使用固定文件名 `infer.py`，内含无参 `main()` 函数
2. main() 函数执行后**必须在 EVAL_RESULT_PATH 目录生成 predictions.json**

##### predictions.json 格式要求
- 顶级字段为 `predictions`
- 结构是：字符串user_id → 0~1之间的预测转化率浮点数

##### 格式示例
```json
{
    "predictions": {
        "user_001": 0.8732,
        "user_002": 0.1245,
        "user_003": 0.5621
    }
}
```

##### 重要提醒
JSON中必须严格匹配测试集所有合法user_id，不能缺失、不能多出多余ID，否则会影响最终评分。