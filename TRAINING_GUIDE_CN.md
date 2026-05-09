# LeRobot 训练与仿真完整指南

> 本文档为 LeRobot 框架的中文训练指南，详细说明代码流程、仿真操作步骤和最佳实践。

# 目录

- [项目概述](#项目概述)
- [代码架构流程](#代码架构流程)
- [训练流程详解](#训练流程详解)
- [仿真环境操作](#仿真环境操作)
- [数据集管理](#数据集管理)
- [策略训练实战](#策略训练实战)
- [常见问题与解决方案](#常见问题与解决方案)

---

## 项目概述

LeRobot 是一个基于 PyTorch 的机器人学习框架，提供：

- **统一的机器人接口**：支持 SO-101、ALOHA、Koch 等多种机器人
- **标准化数据集格式**：Parquet + MP4，支持 HuggingFace Hub
- **先进策略实现**：ACT、Diffusion、VQBeT、TD-MPC 等
- **完整训练工具链**：数据收集、训练、评估、部署

### 核心技术栈

```
Python 3.12+ | PyTorch | HuggingFace Hub | Accelerate | WandB | Gymnasium
```

---

## 代码架构流程

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LeRobot 系统架构                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │  数据层      │    │  策略层      │    │  执行层      │                 │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤                 │
│  │ LeRobotDataset│    │PreTrainedPolicy│    │   Robot     │                 │
│  │ - Parquet    │───→│ - ACT        │───→│ - SO-101    │                 │
│  │ - MP4 视频   │    │ - Diffusion  │    │ - ALOHA     │                 │
│  │ - 元数据     │    │ - VQBeT      │    │ - Koch      │                 │
│  └──────────────┘    └──────────────┘    └──────────────┘                 │
│         ↓                   ↓                   ↓                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │  数据处理    │    │  训练引擎    │    │  仿真环境    │                 │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤                 │
│  │ Processor    │    │ Accelerate   │    │ Gymnasium    │                 │
│  │ - 归一化     │    │ - 分布式训练 │    │ - PushT      │                 │
│  │ - 数据增强   │    │ - 混合精度   │    │ - ALOHA Env  │                 │
│  │ - 特征映射   │    │ - 梯度累积   │    │ - Isaac Lab  │                 │
│  └──────────────┘    └──────────────┘    └──────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 核心模块说明

#### 1. 数据层 (`src/lerobot/datasets/`)

**LeRobotDataset** 是核心数据集类，负责：

```python
# 数据集文件结构
dataset/
├── data/                    # 主数据 (Parquet 格式)
│   └── chunk-XXX/
│       └── file-XXX.parquet
├── meta/                    # 元数据
│   ├── info.json           # 数据集信息 (shapes, keys, fps)
│   ├── stats.json          # 统计数据 (用于归一化)
│   └── tasks.parquet       # 任务描述
└── videos/                  # 视频数据 (MP4 格式)
    └── observation.images.XXX/
```

**关键功能：**

- 时间戳对齐：通过 `delta_timestamps` 加载历史/未来帧
- 视频解码：实时从 MP4 提取帧
- Episode 管理：支持按 episode 加载和采样

#### 2. 策略层 (`src/lerobot/policies/`)

**PreTrainedPolicy** 是所有策略的基类：

```
PreTrainedPolicy (基类)
├── ACTPolicy          # Action Chunking Transformer
├── DiffusionPolicy    # Diffusion Policy
├── VQBeTPolicy        # Vector-Quantized Behavior Transformer
├── TDMPCPolicy        # TD-MPC (Model-based RL)
├── PI0Policy          # Vision-Language-Action Model
└── GrootPolicy        # NVIDIA GR00T
```

**核心方法：**

- `forward(batch)` → 训练前向传播，返回 loss
- `select_action(obs)` → 推理时选择动作
- `reset()` → 重置内部状态
- `get_optim_params()` → 获取优化器参数

#### 3. 训练引擎 (`src/lerobot/scripts/lerobot_train.py`)

**训练流程：**

```
阶段1: 初始化
├── 验证配置有效性
├── 创建 Accelerator (支持分布式训练)
├── 设置随机种子 (保证可复现性)
└── 初始化 WandB 日志 (可选)

阶段2: 数据准备
├── 加载数据集 (从本地或 HuggingFace Hub)
├── 创建数据加载器 (支持多进程加载)
└── 创建 EpisodeAwareSampler (可选，用于序列采样)

阶段3: 模型创建
├── 创建策略网络 (ACT/Diffusion/VQBeT等)
├── 创建预处理器和后处理器 (数据归一化等)
├── 创建优化器和学习率调度器
└── 可选：加载 PEFT 适配器 (LoRA等)

阶段4: 训练循环
for step in range(total_steps):
    ├── 获取数据批次
    ├── 调用 update_policy() 更新模型
    ├── 记录训练指标
    ├── 定期保存检查点
    └── 定期评估策略 (如果配置了环境)

阶段5: 收尾
├── 保存最终模型
└── 推送到 HuggingFace Hub (可选)
```

#### 4. 环境层 (`src/lerobot/envs/`)

**EnvConfig** 是环境配置基类：

```
EnvConfig (基类)
├── AlohaEnv      # ALOHA 双臂机器人仿真
├── PushtEnv      # PushT 推方块任务
├── LiberoEnv     # LIBERO 操作基准测试
├── MetaworldEnv  # MetaWorld 多任务基准
├── RobocasaEnv   # RoboCasa 厨房环境
└── HubEnvConfig  # 从 HuggingFace Hub 加载环境
```

---

## 训练流程详解

### 1. 单步训练更新 (`update_policy`)

```python
def update_policy(policy, batch, optimizer, grad_clip_norm, accelerator):
    """
    执行单步训练更新策略权重
  
    流程:
    1. 前向传播: 计算预测动作和损失
    2. 反向传播: 计算梯度
    3. 梯度裁剪: 防止梯度爆炸
    4. 优化器更新: 更新模型参数
    5. 学习率调度: 调整学习率
    """
    policy.train()
  
    # 1. 前向传播 (混合精度)
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
  
    # 2. 反向传播
    accelerator.backward(loss)
  
    # 3. 梯度裁剪
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
  
    # 4. 优化器更新
    optimizer.step()
    optimizer.zero_grad()
  
    # 5. 学习率调度
    if lr_scheduler is not None:
        lr_scheduler.step()
  
    return loss, output_dict
```

### 2. ACT 策略详解

**Action Chunking Transformer (ACT)** 核心思想：

```
传统方法: 观测 o_t ──→ 策略 π ──→ 单步动作 a_t

ACT 方法: 观测 o_t ──→ 策略 π ──→ 动作序列 [a_t, ..., a_{t+k}]
          └── 一次预测多步，减少累积误差
```

**ACT 架构：**

```
输入观测
├── 图像观测 ──→ ResNet Backbone ──→ 视觉特征
└── 机器人状态 ──→ 状态编码器 ──→ 状态特征

VAE 编码器 (可选)
└── 随机潜变量 z ~ N(μ, σ²) ──→ 增强多模态数据表现

Transformer Decoder
├── 视觉特征 + 状态特征 + z ──→ Cross Attention
└── 自回归解码 ──→ 动作序列 [a_t, a_{t+1}, ..., a_{t+chunk_size}]

输出: chunk_size 步动作
```

**关键参数：**

- `chunk_size = 100`: 每次预测 100 步动作
- `n_action_steps = 8`: 每次执行 8 步后重新预测
- `use_vae = True`: 是否使用 VAE
- `temporal_ensemble_coeff`: 时序集成系数 (可选)

---

## 仿真环境操作

### 1. 环境创建与使用

```python
from lerobot.envs import make_env, AlohaEnv

# 方式1: 使用配置类
env_cfg = AlohaEnv(task="AlohaInsertion-v0", fps=50)
envs = env_cfg.create_envs(n_envs=4)

# 方式2: 使用工厂函数
envs = make_env(env_cfg, n_envs=4, use_async_envs=True)

# 与环境交互
obs, info = envs.reset()
action = policy.select_action(obs)
obs, reward, done, truncated, info = envs.step(action)
```

### 2. 支持的仿真环境

#### PushT (入门级)

```bash
# 训练 PushT 任务
lerobot-train \
    --policy.path=lerobot/act_pusht \
    --dataset.repo_id=lerobot/pusht \
    --save_freq=10000 \
    --eval_freq=20000
```

**特点：**

- 简单的 2D 推方块任务
- 适合快速验证策略
- 训练时间短 (约 1-2 小时)

#### ALOHA (双臂操作)

```bash
# 训练 ALOHA 任务
lerobot-train \
    --policy.path=lerobot/act_aloha_sim_insertion \
    --dataset.repo_id=lerobot/aloha_sim_insertion \
    --save_freq=20000 \
    --eval_freq=20000
```

**特点：**

- 双臂协作任务
- 支持插入、转移等操作
- 需要更多训练数据

### 3. Isaac Lab 集成 (高级)

Isaac Lab 是 NVIDIA 的高性能仿真平台，支持：

```python
# Isaac Lab 环境配置 (需要额外安装)
from lerobot.envs import IsaacLabEnv

env_cfg = IsaacLabEnv(
    task="Isaac-Lift-Cube-Franka-v0",
    fps=30,
    headless=True,  # 无头模式，适合服务器训练
    num_envs=1024,  # 并行环境数量
)
```

**安装步骤：**

```bash
# 1. 安装 Isaac Sim (需要 NVIDIA GPU)
# 参考: https://developer.nvidia.com/isaac-sim

# 2. 安装 Isaac Lab
pip install isaac-lab

# 3. 安装 LeRobot 的 Isaac Lab 扩展
pip install 'lerobot[isaac_lab]'
```

---

## 数据集管理

### 1. 加载数据集

```python
from lerobot.datasets import LeRobotDataset

# 从 HuggingFace Hub 加载
dataset = LeRobotDataset("lerobot/pusht")

# 从本地加载
dataset = LeRobotDataset("my_dataset", root="./data")

# 加载特定 episodes
dataset = LeRobotDataset("lerobot/pusht", episodes=[0, 1, 2])

# 加载带时间偏移的数据
delta_timestamps = {
    "action": [-0.1, 0, 0.1],  # 加载前后 0.1 秒的动作
}
dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
```

### 2. 创建数据集

```python
from lerobot.datasets import LeRobotDataset

# 创建空数据集
dataset = LeRobotDataset.create(
    repo_id="my_new_dataset",
    fps=30,
    features={
        "observation.image": {"shape": [3, 480, 640], "dtype": "video"},
        "observation.state": {"shape": [14], "dtype": "float32"},
        "action": {"shape": [14], "dtype": "float32"},
    }
)

# 添加帧
dataset.add_frame({
    "observation.image": image,
    "observation.state": state,
    "action": action,
})

# 保存 episode
dataset.save_episode()
```

### 3. 数据集可视化

```bash
# 可视化数据集
lerobot-dataset-viz --repo-id lerobot/pusht

# 转换数据集格式
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id my_old_dataset
```

---

## 策略训练实战

### 1. 快速开始

```bash
# 使用预训练配置训练
lerobot-train \
    --policy.path=lerobot/act_pusht \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=outputs/train/pusht_act \
    --batch_size=8 \
    --steps=100000 \
    --save_freq=20000 \
    --eval_freq=20000 \
    --wandb.enable=true \
    --wandb.project=lerobot-training
```

### 2. 自定义训练配置

```python
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.datasets import DatasetConfig

# 创建自定义配置
config = TrainPipelineConfig(
    dataset=DatasetConfig(
        repo_id="my_custom_dataset",
        root="./data",
    ),
    policy=ACTConfig(
        chunk_size=100,
        n_action_steps=8,
        use_vae=True,
        kl_weight=10.0,
    ),
    batch_size=16,
    steps=200000,
    save_freq=10000,
    eval_freq=20000,
    wandb=WandBConfig(
        enable=True,
        project="my-project",
    ),
)

# 开始训练
from lerobot.scripts.lerobot_train import train
train(config)
```

### 3. 分布式训练

```bash
# 使用 Accelerate 启动分布式训练
accelerate config  # 配置分布式设置

# 单机多 GPU
accelerate launch \
    $(which lerobot-train) \
    --policy.path=lerobot/act_aloha \
    --dataset.repo_id=lerobot/aloha_sim_insertion \
    --batch_size=32

# 多机多 GPU
accelerate launch \
    --num_processes=8 \
    --num_machines=2 \
    --machine_rank=0 \
    --main_process_ip=192.168.1.1 \
    $(which lerobot-train) \
    --policy.path=lerobot/act_aloha \
    --dataset.repo_id=lerobot/aloha_sim_insertion
```

### 4. 恢复训练

```bash
# 从检查点恢复训练
lerobot-train \
    --config_path=outputs/train/pusht_act/2024-01-01/12-00-00_train_config.json \
    --resume=true
```

### 5. 评估策略

```bash
# 评估训练好的策略
lerobot-eval \
    --policy.path=outputs/train/pusht_act/checkpoints/100000/pretrained_model \
    --env.task=AlohaInsertion-v0 \
    --eval.batch_size=10 \
    --eval.n_episodes=50
```

---

## 常见问题与解决方案

### 1. 训练相关

**Q: 训练时显存不足怎么办？**

```bash
# 方案1: 减小 batch_size
lerobot-train --batch_size=4

# 方案2: 使用梯度累积
lerobot-train --batch_size=4 --gradient_accumulation_steps=4

# 方案3: 使用混合精度训练
lerobot-train --use_amp=true
```

**Q: 训练速度太慢？**

```bash
# 方案1: 增加数据加载进程
lerobot-train --num_workers=8

# 方案2: 使用预取
lerobot-train --prefetch_factor=4

# 方案3: 使用更快的数据存储 (SSD)
```

**Q: 如何选择合适的策略？**

```
简单任务 (PushT, 单臂操作):
└── ACT 或 Diffusion Policy

复杂任务 (双臂协作, 长序列):
└── ACT (推荐) 或 VQBeT

需要在线学习:
└── TD-MPC 或 SAC

视觉-语言-动作任务:
└── PI0 或 Groot
```

### 2. 数据相关

**Q: 如何处理视频解码错误？**

```python
# 使用不同的视频后端
dataset = LeRobotDataset(
    "my_dataset",
    video_backend="pyav",  # 或 "torchcodec"
)
```

**Q: 如何合并多个数据集？**

```python
from lerobot.datasets import LeRobotDataset

# 合并数据集
dataset1 = LeRobotDataset("dataset1")
dataset2 = LeRobotDataset("dataset2")

# 使用 multi_dataset
from lerobot.datasets import MultiDataset
multi_dataset = MultiDataset([dataset1, dataset2])
```

### 3. 环境相关

**Q: 如何创建自定义环境？**

```python
from lerobot.envs import EnvConfig
from dataclasses import dataclass

@dataclass
class MyCustomEnv(EnvConfig):
    task: str = "MyTask-v0"
    fps: int = 30
  
    @property
    def gym_kwargs(self) -> dict:
        return {
            "max_episode_steps": 500,
        }
  
    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        import gymnasium as gym
        # 创建你的环境
        return gym.make(self.gym_id, **self.gym_kwargs)
```

### 4. 部署相关

**Q: 如何在真实机器人上部署？**

```python
from lerobot.robots import MyRobot
from lerobot.policies import ACTPolicy

# 加载策略
policy = ACTPolicy.from_pretrained("my_trained_policy")
policy.eval()

# 连接机器人
robot = MyRobot(config=...)
robot.connect()

# 控制循环
while True:
    obs = robot.get_observation()
    action = policy.select_action(obs)
    robot.send_action(action)
```

---

## 最佳实践

### 1. 数据收集

- **数据质量 > 数据数量**：确保遥操作数据质量高
- **多样性**：收集不同场景、光照、物体的数据
- **标注**：为每个 episode 添加任务描述
- **验证**：定期检查数据集质量和一致性

### 2. 训练技巧

- **从小任务开始**：先在 PushT 上验证流程
- **监控训练**：使用 WandB 跟踪 loss、学习率等指标
- **定期评估**：在仿真环境中测试策略性能
- **保存检查点**：定期保存，避免训练中断丢失进度

### 3. 超参数调优

```bash
# 使用 WandB Sweep 进行超参数搜索
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

**关键超参数：**

- `learning_rate`: 1e-4 到 1e-5
- `batch_size`: 8 到 32
- `chunk_size`: 100 (ACT)
- `n_action_steps`: 8 (ACT)

### 4. 性能优化

- **使用 SSD**：将数据集放在 SSD 上
- **预加载视频**：`download_videos=True`
- **调整数据加载器**：`num_workers`, `prefetch_factor`
- **混合精度训练**：自动启用，无需额外配置

---

## 参考资源

- [LeRobot 官方文档](https://huggingface.co/docs/lerobot)
- [HuggingFace Hub 数据集](https://huggingface.co/lerobot)
- [Discord 社区](https://discord.gg/q8Dzzpym3f)
- [GitHub Issues](https://github.com/huggingface/lerobot/issues)

---

## 更新日志

- 2026-05-06: 创建中文训练指南，添加代码流程和仿真操作步骤
- 包含完整的训练、评估、部署流程说明

---

**祝你训练顺利！如有问题，欢迎在 GitHub Issues 或 Discord 社区提问。**
