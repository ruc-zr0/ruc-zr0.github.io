# ZR-0: Pre-Training Generalist Vision–Language–Action Model with Embodied Chain-of-Thought Reasoning

This repository releases **ZR-0**, a large-scale foundation **Vision-Language-Action (VLA)** model for general-purpose robotic manipulation.

ZR-0 is **pre-trained on a diverse mixture of cross-embodiment manipulation data and general vision–language corpora** (e.g., VQA and image captioning). The pre-training dataset comprises **over 400k trajectories** collected from a wide range of robot embodiments, including **Franka, xArm, GR-1, ALOHA, ARX5, UR5**, and others, covering diverse environments and execution styles.

Most importantly, **each frame in the pre-training robotic datasets is annotated with a _Embodied Chain-of-Thought (ECoT)_ reasoning path**, providing explicit and grounded multi-step decision traces, as illustrated below:
<p align="center">
  <img src="./images/embodied_cot.png" alt="Architecture"><br>
  <em>An Example of Embodied Chain-of-Thought Reasoning in ZR-0's Pre-training Datasets.</em>
</p>

ZR-0 is co-trained with a **next-token prediction loss for ECoT reasoning** and a **flow-matching loss for continuous action chunk generation**, enabling joint learning of reasoning and control. These embodied reasoning signals act as a strong supervisory bridge between **language reasoning, task planning, spatial perception, and low-level action generation**. As a result, ZR-0 not only demonstrates strong **zero-shot performance** on settings seen during pre-training, but also exhibits **high adaptability to novel robot embodiments, environments, tasks, and behaviors** through lightweight post-training.

ZR-0 contains **2.6 billion parameters** and follows a **System 2 + System 1** hybrid design:
- **System 2**: a powerful vision–language backbone based on **Qwen3-VL-2B**
- **System 1**: a **flow-matching-based action expert** for continuous action generation

Given a language instruction, visual observations, and robot state, ZR-0 directly outputs **a chunk of executable actions**. The overall architecture of ZR-0 is illustrated below.
<p align="center">
  <img src="./images/model_architecture.png" alt="Architecture"><br>
  <em>ZR-0's Model Architecture.</em>
</p>

## 安装Lerobot

**第一步**: 安装lerobot
```sh
# 创建新anaconda环境
conda create -y -n ZR0 python=3.10
conda activate ZR0

# 进入lerobot目录
cd lerobot
# 在环境中安装lerobot源码
pip install -e .
# 返回项目root路径
cd ..
```

**第二步**： 安装模型训练和部署推理所需的包
```sh
# 激活环境
conda activate ZR0
# 安装训练和部署模型需要的包
pip install -r requirements.txt
# 单独安装Flash Attention
# 需要自己去 https://github.com/Dao-AILab/flash-attention/releases 找合适的版本下载编译好的whl文件，以下是一个示例：CUDA 12 (cu12) + torch 2.6 (torch2.6) + python 3.10 (cp310)
pip install flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 模型训练
训练模型涉及以下几个关键的文件/文件夹：

- `scripts/run_train_vla_1_node-example.sh`是一个用于启动单机8卡分布式训练的脚本，里面配置了很多关键的参数，比如VLM模型文件目录和FAST tokenizer的文件目录、epochs，batch size，learning rate，dataset path（这里以LIBERO举例）等等。配置完成后直接执行`sh scripts/run_train_vla_1_node-example.sh`即可。参数的详细说明请参阅`train_vla.py`文件。（注意，vlm_name_or_path和FAST_tokenizer_path需要按照自己机器的实际目录进行配置，他们可以在 https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct 和 https://huggingface.co/physical-intelligence/fast 下载）
- `accelerate_config_vla_1_node.yaml`是accelerate分布式训练的配置文件，我们目前默认采用DeepSpeed Zero-2 + bfloat16来节省训练时显存开销。
- `train_vla.py`是训练脚本。
- `model`文件夹实现了模型结构。
- `utils`文件夹实现了一些关键的工具组件。注意，如果您想添加一个新数据集，请往`utils/constants.py`的`DATASET2FEATURE`中添加对应的数据集元信息。
- `outputs`文件夹包含了tensorboard的log以及训练过程中保存的检查点。

我用的单节点的集群入口命令为:
```sh
source /root/anaconda3/etc/profile.d/conda.sh
conda activate ZR0

# set CUDA
export CUDA_HOME=/root/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# set HF_HOME cache dir
export HF_HOME="/root/.hf_cache"
# set HF_LEROBOT_HOME cache dir
export HF_LEROBOT_HOME="/root/.lerobot_cache/lerobot"

cd /root/ZR0
sh ./scripts/run_train_vla_1_node-example.sh
```

集群环境变量为：
| 变量名                | 值        |
|-----------------------|-----------|
| MLP_DEBUG_CLEANUP     | 1         |
| MLP_SOCKET_IFNAME     | bond0     |
| MLP_SKIP_SORT_RDMA    | true      |
| NCCL_SOCKET_IFNAME    | bond0     |

## 模型部署推理
我们现在所有的VLA部署都采用server和client分离的策略，他们中间通过websocket进行双向通信。
- server端负责接受task，images，state等信息，调用模型生成action chunk传给client端。
- client端负责和机械臂环境做交互（真机或者模拟环境），将action chunk发送到环境中执行，记录执行状态并从环境中读取最新的observation（包括images，state等）发给server。

server端的统一接口在`server.py`脚本下，以下是一个启动的示例：
```sh
CUDA_VISIBLE_DEVICES=4 python server.py --env_type libero --ckpt_dir your-ckpt-path --port 8001
```
启动的时候需要指定具体环境是什么（这里env_type为libero）。

client端因为涉及不同环境，因此需要些不同的交互脚本，本项目提供了一个libero的示例，详见`evaluation/libero_eval/run_libero_eval.py`文件。以下是一个启动的示例：
```sh
python -m evaluation.libero_eval.run_libero_eval --args.task-suite-name libero_spatial --args.port 8001
```
启动的时候需要指定server的port，保持一致。

注意：如需评估libero，请参阅组内文档来配置libero所需环境（这个anaconda环境独立于本项目使用的环境）：https://zhipu-ai.feishu.cn/docx/CUq6dDxBpoid41xbYgDcABTvnf7