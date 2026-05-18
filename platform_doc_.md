
首页
LeAsk
AI Knowledge Hub
全部
搜索
avatar
乐问 问题详情 我要提问
AngelML
  修改标签
Tencent Angel Machine Learning Platform User Guide
 Alex 04/25 19:53 浏览(2744) 回答(3)
User Guidance


Model Training




Create and submit a training task
1.  Fill in "Job Name".

2.  Fill in "Job Description".

3.  Click "Local Upload" to upload scripts from local files OR click "New Script" to start writing a new script.

 

ATTENTION: The training template includes a mandatory run.sh file as the execution entry point, which will be executed automatically at task startup!

 

4.  After confirmation, click "Submit" to save the task.



Run a task
Click "Run" to start running a task.



Instance status check
After you click "Run", the "Instance Status" automatically changes to "Training Job Running".



Output and Logs check:
Select the task to check and click "Instances". You will be directed to a page containing "Output" and "Logs", both of which can be clicked for more details. You could also click "Stop" to terminate the running task.





Copy Delete Edit example
Click "More" to copy, delete, or edit current task.



Script example


 

Model Release And Management
Model release steps
1.  Click "Instances" from the training task.



2.  Click "Output".



3.  Select the ckpt you want to release, click "Publish" and fill in the "Model Name" and "Model Description", after confirmation click "Submit".



4.  "Publish Status" will automatically switch to "Released" if submitted successfully.



5.  Click "Model Management" (left sidebar) to review and manage released models.



Model edit or delete


 

Model Evaluation




 

Entrance
Method 1: On the "Model Management" page, locate the model you want to evaluate and click the "Model Evaluation" button at the bottom of its card.



Method 2: On the "Model Evaluation" page, click "Create Evaluation".



Submit evaluation task
● Steps

1.  Select Model: Choose the trained model you want to evaluate from the dropdown list.

2.  Upload Inference Code: Upload your inference scripts via “Upload from Local” or create new scripts via “New Script”.

● You must include an entry script named "infer.py", which must contain a main() function that takes no arguments.

● You may also upload other supporting scripts (e.g., "dataset.py", "model.py", or any custom modules) that your "infer.py" imports. All uploaded files will be placed in the EVAL_INFER_PATH directory.

● The total size of all uploaded scripts is limited to 100 MB.

3.  Submit: Click the “Submit” button to create the evaluation task.

 

ATTENTION: 

1.  The script must be strictly named "infer.py" and must contain a main() function that takes no arguments, otherwise the evaluation would fail!

2.  The daily submission limit resets every 24 hours at 15:59:59 AOE time. Each team may submit up to 3 evaluation tasks per rolling 24‑hour period（as the  Industrial Track raised to  4  since  April 30, 2026, 16:00 AOE）. Failed or stopped tasks do not count toward the daily limit.

3.  A maximum time limit of 30 minutes is imposed on each inference task in both the Industry and Academic tracks. Any inference job that exceeds this time limit will be automatically terminated by the system.

 



 



● Evaluation Status

Once submitted, the evaluation task will go through the following stages:

Status

Description

Pending

The task has been submitted and is queued.

Waiting for Inference Resources

Waiting for available compute resources.

Inference Running

Your "infer.py" script is executing.

Waiting for Evaluation Resources

Inference completed. Waiting for scoring resources.

Evaluation Running

The platform is scoring your predictions.

Success

Evaluation completed. You can view your score.

Failed

Evaluation failed. Check the logs for details.



 

Install dependencies
● If you wish to execute shell commands to install dependencies — similar to the "run.sh" script used during training — you can upload a "prepare.sh" script (which must be named prepare.sh) when creating an evaluation task. This script runs automatically before inference starts, directly within the pre-activated Conda environment. A template is provided below:

 

#!/bin/bash
 
# 1. This script will be executed before the inference logic starts.
# 2. This script runs inside the activated conda environment.
# 3. You can execute your own preparation commands here, such as:
#      conda install <package>
#      or any other setup commands you need.
 

 

Platform Specifications
Operating Environment
Model training and evaluation run in a consistent hardware and software environment.

1.  Hardware Environment
The platform deploys high-performance GPUs with virtualized resource allocation. Each GPU partition meets the following specifications:

Resource

Specification

Computing Power

20% of a single GPU

GPU Memory

19GiB

CPU Cores

9

Memory

55GiB

2.  Software Environment
● The operating system is Ubuntu 22.04. The following foundational software packages are pre-installed:

Software

Version

cuda

12.6

cudnn

9.5.1

cublas

12.6.3.3

nccl

2.26.2+cuda12.6

conda

26.1.1

python

3.10.20

● Installed Python packages

 

Package                      Version
---------------------------- ----------------
absl-py                      2.4.0
accelerate                   0.32.1
aiohappyeyeballs             2.6.1
aiohttp                      3.13.3
aiosignal                    1.4.0
albucore                     0.0.24
albumentations               2.0.8
alembic                      1.18.4
annotated-doc                0.0.4
annotated-types              0.7.0
astunparse                   1.6.3
async-timeout                5.0.1
attrs                        25.4.0
audioread                    3.1.0
beautifulsoup4               4.14.3
bitsandbytes                 0.49.2
blis                         1.3.3
Bottleneck                   1.4.2
catalogue                    2.0.10
catboost                     1.2.10
certifi                      2026.2.25
cffi                         2.0.0
charset-normalizer           3.4.6
click                        8.3.1
cloudpathlib                 0.23.0
cloudpickle                  3.1.2
colorlog                     6.10.1
confection                   0.1.5
contourpy                    1.3.1
cycler                       0.12.1
cymem                        2.0.13
datasets                     2.14.7
decorator                    5.2.1
dill                         0.3.7
duckdb                       1.4.4
einops                       0.8.2
en_core_web_sm               3.8.0
fbgemm_gpu                   1.2.0+cu126
filelock                     3.25.2
flatbuffers                  25.12.19
fonttools                    4.61.0
frozenlist                   1.8.0
fsspec                       2023.10.0
gast                         0.7.0
gensim                       4.4.0
google-pasta                 0.2.0
graphviz                     0.21
greenlet                     3.3.2
grpcio                       1.78.0
h5py                         3.16.0
huggingface-hub              0.17.3
idna                         3.11
ImageIO                      2.37.3
importlib_metadata           8.7.1
iopath                       0.1.10
Jinja2                       3.1.6
joblib                       1.5.3
jsonlines                    4.0.0
keras                        3.12.1
kiwisolver                   1.4.9
lazy-loader                  0.5
libclang                     18.1.1
librosa                      0.11.0
lightgbm                     4.6.0
lightning-utilities          0.15.3
llvmlite                     0.46.0
Mako                         1.3.10
Markdown                     3.10.2
markdown-it-py               4.0.0
MarkupSafe                   3.0.3
matplotlib                   3.10.8
mdurl                        0.1.2
mkl_fft                      2.1.1
mkl_random                   1.3.0
mkl-service                  2.5.2
ml_dtypes                    0.5.4
mpmath                       1.3.0
msgpack                      1.1.2
multidict                    6.7.1
multiprocess                 0.70.15
murmurhash                   1.0.15
mypy_extensions              1.1.0
namex                        0.1.0
narwhals                     2.18.0
networkx                     3.4.2
nltk                         3.9.3
numba                        0.64.0
numexpr                      2.14.1
numpy                        2.2.5
opencv-python                4.13.0.92
opencv-python-headless       4.13.0.92
opt_einsum                   3.4.0
optree                       0.19.0
optuna                       4.8.0
orjson                       3.11.7
packaging                    25.0
pandas                       2.3.3
patsy                        1.0.2
peft                         0.13.2
pillow                       12.1.1
pip                          26.0.1
platformdirs                 4.9.4
plotly                       6.6.0
polars                       1.39.2
polars-runtime-32            1.39.2
pooch                        1.9.0
portalocker                  3.2.0
preshed                      3.0.12
propcache                    0.4.1
protobuf                     5.29.6
psutil                       7.2.2
pyarrow                      23.0.1
pyarrow-hotfix               0.7
pycparser                    3.0
pydantic                     2.12.5
pydantic_core                2.41.5
Pygments                     2.19.2
pyparsing                    3.2.5
PyQt6                        6.10.2
PyQt6_sip                    13.11.0
pyre-extensions              0.0.32
python-dateutil              2.9.0.post0
pytorch-lightning            2.6.1
pytz                         2026.1.post1
pyvers                       0.2.2
PyYAML                       6.0.3
regex                        2026.2.28
requests                     2.32.5
rich                         14.3.3
sacremoses                   0.1.1
safetensors                  0.7.0
scikit-image                 0.25.2
scikit-learn                 1.7.2
scipy                        1.15.3
seaborn                      0.13.2
sentencepiece                0.2.1
setuptools                   80.10.2
shellingham                  1.5.4
simsimd                      6.5.16
sip                          6.15.1
six                          1.17.0
smart_open                   7.5.1
soundfile                    0.13.1
soupsieve                    2.8.3
soxr                         1.0.0
spacy                        3.8.11
spacy-legacy                 3.0.12
spacy-loggers                1.0.5
SQLAlchemy                   2.0.48
srsly                        2.5.2
statsmodels                  0.14.6
stringzilla                  4.6.0
sympy                        1.14.0
tensorboard                  2.19.0
tensorboard-data-server      0.7.2
tensordict                   0.11.0
tensorflow                   2.19.0
tensorflow-io-gcs-filesystem 0.37.1
termcolor                    3.3.0
textblob                     0.19.0
thinc                        8.3.10
threadpoolctl                3.5.0
tifffile                     2025.5.10
timm                         1.0.25
tokenizers                   0.14.1
tomli                        2.4.0
torch                        2.7.1+cu126
torch_cluster                1.6.3+pt27cu126
torch-geometric              2.7.0
torch_scatter                2.1.2+pt27cu126
torch_sparse                 0.6.18+pt27cu126
torchaudio                   2.7.1+cu126
torchmetrics                 1.0.3
torchrec                     1.2.0+cu126
torchtext                    0.18.0
torchvision                  0.22.1+cu126
tornado                      6.5.4
tqdm                         4.67.3
transformers                 4.35.0
triton                       3.3.1
typer                        0.24.1
typer-slim                   0.24.0
typing_extensions            4.15.0
typing-inspect               0.9.0
typing-inspection            0.4.2
tzdata                       2025.3
urllib3                      2.6.3
wasabi                       1.1.3
weasel                       0.4.3
Werkzeug                     3.1.6
wheel                        0.46.3
wrapt                        2.1.2
xgboost                      3.2.0
xxhash                       3.6.0
yarl                         1.23.0
zipp                         3.23.0
 

● To install other packages, use conda or pip3.

For instance: to install a package like pandas (Note: pandas is pre-installed in the default environment; this is just an example).

 

# use conda
conda install -y pandas
 
# use pip3
pip3 install pandas
 

Model Training:
1.  Environment Variables
The platform provides directories required for training (e.g., training datasets, outputs, logs, etc.), and passes them into the container as environment variables. You can read these variables in your script.

Variable Name

Description

USER_CACHE_PATH

User cache storage path (quota: 20GB). 

This variable is provided for both training and evaluation, allowing you to share files between these two stages.

TRAIN_DATA_PATH

Path to training datasets.

TRAIN_CKPT_PATH

Path for saving model checkpoints. 

TRAIN_TF_EVENTS_PATH

Path for TensorBoard event files.

In a shell script, you can read environment variables as follows:

 

${TRAIN_DATA_PATH}
 

In a Python script, you can read environment variables as follows：

 

import os
os.environ.get("TRAIN_DATA_PATH")
 

2.  Output Specifications
2.1. Model Output

a. Model weights must be saved to the directory specified by TRAIN_CKPT_PATH.

b. Iterative checkpoint files must be placed in a directory prefixed with global_step; otherwise, the platform will not recognize them. The directory name must not exceed 300 characters.

c. Directory names may only contain letters (a-z, A-Z), numbers (0-9), underscores (_), hyphens (-), equal signs (=), and periods (.).

 

To include additional parameter information in the directory name, append it after "global_step".

For example, a checkpoint from the 20th iteration with learning rate (lr=0.001), layer count (layer=2), head count (head=1), hidden size (hidden=50), and max sequence length (maxlen=200) should be saved under: "global_step20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200"

 

2.2. TensorBoard Metrics

 

For PyTorch TensorBoard usage, refer to:https://docs.pytorch.org/docs/2.7/tensorboard.html#

 

As shown in the following code, you can read the "TRAIN_TF_EVENTS_PATH" environment variable to initialize a SummaryWriter. The platform only supports scalar metrics.

 

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
 

Model Evaluation:
1.Environment Variables
The platform provides directories needed during the evaluation process (such as evaluation datasets, outputs etc.) and passes them into the container as environment variables. You can read these variables in your script.

Variable Name

Description

USER_CACHE_PATH

User cache path, quota 20GB.

This variable is provided for both training and evaluation, allowing you to share files between these two stages.

MODEL_OUTPUT_PATH

Model output path.

EVAL_DATA_PATH

Path to test data directory for inference.

EVAL_RESULT_PATH

Output path for intermediate results and "predictions.json".

EVAL_INFER_PATH

Directory for user-uploaded inference script files.

In a shell script, you can read environment variables as follows:

 

${EVAL_DATA_PATH}
 

In a Python script, you can read environment variables as follows：

 

import os
os.environ.get("EVAL_DATA_PATH")
 

2.  Output Specifications
The script must be strictly named "infer.py" and must contain a main() function that takes no arguments, otherwise the evaluation would fail!

The main() function must produce a "predictions.json" file and save it under the EVAL_RESULT_PATH directory.

 

● predictions.json Format

The JSON file must contain a predictions field, which is a mapping from user_id (string) to the predicted conversion probability (float, range 0–1).

Field

Data Format

Description

predictions

map<string, float>

A mapping where each key is a user_id (string) and each value is the model’s predicted probability that the user’s interaction is positive (i.e., a conversion).

 

● Example of predictions.json:

 

{
    "predictions": {
        "user_001": 0.8732,
        "user_002": 0.1245,
        "user_003": 0.5621
    }
}
 

 

Attention：Each key in the predictions map must be a valid user_id from the test dataset, and the corresponding value must be the model’s predicted conversion probability for that user’s interaction. Missing or extra user_ids may affect the final score!

 


 邀请回答  引用问题  二维码
 关注 ( 0 )
 
我要回答
3个回答
默认排序 

赞同
Marlesson Santana  Contestant Group\Production Group\Industrial\MS-SOLO
5人赞同
How can we access Tencent Angel Machine Learning Platform User Guide without "WeCom"?

评论(0) 引用04/26 18:37

赞同
Alex  Administrators
1人赞同
Since the competition has already begun, we highly recommend that all participants review the official User Guide. This manual contains essential operational workflows and technical instructions to help you navigate the platform and optimize your performance.

You can access the document via the following link:

Tencent Angel Machine Learning Platform User Guide

Good luck with the competition!

评论(0) 引用04/25 20:19

赞同
Chris Deotte Contestant Group\Production Group\Industrial\Agent_Recsys
Hi. I submitted the provided sample submission (without changing anything) and it failed. 

![]( https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Apr-2026/e1.png )

And then when i view the logs, the logs are empty. Any advice on how to submit the sample submission successfully?

 

UPDATE: The problem may be that the sample submission code doesn't set the `TRAIN_LOG_PATH` variable. Trying this now.

UPDATE2: Fixing variable above solved first error and now it begins training and makes a log. However now there is an OOM error. Fixing that now...

评论(0) 引用04/28 00:00
已显示全部回答












Copyright©2016-2026 Tencent. All Rights Reserved. 腾讯公司 版权所有
开放接口文档 帮助中心  语言设置 
小程序