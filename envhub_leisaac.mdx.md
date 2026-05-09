# LeIsaac × LeRobot EnvHub 中文指南

> LeRobot EnvHub 现已支持使用 LeIsaac 进行仿真环境中的模仿学习。启动日常操作任务、遥操作机器人、收集演示数据、推送到 Hub，并在 LeRobot 中训练策略 —— 全部在一个循环中完成。

---

## 目录

- [概述](#概述)
- [可用环境](#可用环境)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [随机动作测试](#随机动作测试)
- [遥操作](#遥操作)
- [云端仿真 (无需 GPU)](#云端仿真-无需-gpu)
- [其他说明](#其他说明)

---

## 概述

[LeIsaac](https://github.com/LightwheelAI/leisaac) 集成 IsaacLab 和 SO101 Leader/Follower 设置，提供以下功能：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LeIsaac 核心特性                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  🕹️ 遥操作优先工作流                                                         │
│     └── 用于数据收集的完整遥操作支持                                          │
│                                                                             │
│  📦 内置数据转换                                                              │
│     └── 直接兼容 LeRobot 训练格式                                            │
│                                                                             │
│  🤖 日常技能任务                                                             │
│     ├── 拾取橙子 (Pick Orange)                                               │
│     ├── 抬立方体 (Lift Cube)                                                │
│     ├── 清洁玩具桌 (Clean Toy Table)                                         │
│     └── 折叠布料 (Fold Cloth)                                               │
│                                                                             │
│  ☁️ 持续更新 (来自 LightWheel)                                               │
│     ├── 云端仿真                                                            │
│     ├── EnvHub 支持                                                         │
│     └── Sim2Real 工具链                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 可用环境

下表列出了 LeIsaac x LeRobot EnvHub 中所有可用的任务和环境。也可以通过运行以下命令获取最新的环境列表：

```bash
python scripts/environments/list_envs.py
```

| 任务视频 | 环境 ID | 任务描述 | 相关机器人 |
| :--- | :--- | :--- | :--- |
| 🎬 | [LeIsaac-SO101-PickOrange-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/pick_orange/pick_orange_env_cfg.py)<br><br>[LeIsaac-SO101-PickOrange-Direct-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/pick_orange/direct/pick_orange_env.py) | 拾取三个橙子放入盘子，然后机械臂复位。 | 单臂 SO101 Follower |
| 🎬 | [LeIsaac-SO101-LiftCube-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/lift_cube/lift_cube_env_cfg.py)<br><br>[LeIsaac-SO101-LiftCube-Direct-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/lift_cube/direct/lift_cube_env.py) | 抬起红色立方体。 | 单臂 SO101 Follower |
| 🎬 | [LeIsaac-SO101-CleanToyTable-v0](br><br>[LeIsaac-SO101-CleanToyTable-BiArm-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/clean_toy_table/clean_toy_table_bi_arm_env_cfg.py)<br><br>[LeIsaac-SO101-CleanToyTable-BiArm-Direct-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/clean_toy_table/direct/clean_toy_table_bi_arm_env.py) | 将两个字母 e 物体放入盒子，然后机械臂复位。 | 单臂/双臂 SO101 Follower |
| 🎬 | [LeIsaac-SO101-FoldCloth-BiArm-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/fold_cloth/fold_cloth_bi_arm_env_cfg.py)<br><br>[LeIsaac-SO101-FoldCloth-BiArm-Direct-v0](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/fold_cloth/direct/fold_cloth_bi_arm_env.py) | 折叠布料，然后机械臂复位。<br><br>*注意：此任务仅 DirectEnv 支持 check_success。* | 双臂 SO101 Follower |

---

## 快速开始

### 环境安装

运行以下命令设置代码环境：

```bash
# 参考 Getting Started/Installation 首先安装 leisaac
conda create -n leisaac_envhub python=3.11
conda activate leisaac_envhub

# 安装 CUDA 工具包
conda install -c "nvidia/label/cuda-12.8.1" cuda-toolkit

# 安装 PyTorch
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 安装 leisaac 和 isaaclab
pip install 'leisaac[isaaclab] @ git+https://github.com/LightwheelAI/leisaac.git#subdirectory=source/leisaac' --extra-index-url https://pypi.nvidia.com

# 安装 lerobot
pip install lerobot==0.4.1

# 修复 numpy 版本
pip install numpy==1.26.0
```

### 环境架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LeIsaac + LeRobot 系统架构                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LeIsaac (Isaac Lab 仿真)                                                  │
│   ├── SO101 机械臂仿真                                                       │
│   ├── 视觉渲染 (RGB 相机)                                                   │
│   └── 物理模拟 (MuJoCo/PhysX)                                               │
│           ↓                                                                 │
│   LeRobot EnvHub 包装层                                                     │
│   ├── 统一 Gymnasium 接口                                                   │
│   ├── LeRobot 格式转换                                                      │
│   └── 数据收集支持                                                          │
│           ↓                                                                 │
│   LeRobot 训练                                                              │
│   ├── 数据集创建                                                            │
│   ├── 策略训练                                                              │
│   └── 评估部署                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 使用方法

EnvHub 将每个 LeIsaac 支持的任务暴露在统一的接口中。以下示例加载 `so101_pick_orange` 并演示随机动作 rollout 和交互式遥操作。

### 支持的任务列表

```python
# 可用的任务环境
"so101_pick_orange"       # 拾取橙子
"so101_lift_cube"         # 抬立方体  
"so101_clean_toytable"    # 清洁玩具桌
"bi_so101_fold_cloth"     # 双臂折叠布料

# 加载方式
from lerobot.envs import make_env

# 拾取橙子
envs = make_env(
    "LightwheelAI/leisaac_env:envs/so101_pick_orange.py",
    n_envs=1,
    trust_remote_code=True
)

# 抬立方体
envs = make_env(
    "LightwheelAI/leisaac_env:envs/so101_lift_cube.py",
    n_envs=1,
    trust_remote_code=True
)

# 清洁玩具桌
envs = make_env(
    "LightwheelAI/leisaac_env:envs/so101_clean_toytable.py",
    n_envs=1,
    trust_remote_code=True
)

# 折叠布料 (双臂)
envs = make_env(
    "LightwheelAI/leisaac_env:envs/bi_so101_fold_cloth.py",
    n_envs=1,
    trust_remote_code=True
)
```

---

## 随机动作测试

### 代码示例

```python
# envhub_random_action.py

import torch
from lerobot.envs import make_env

# 从 Hub 加载环境
envs_dict = make_env(
    "LightwheelAI/leisaac_env:envs/so101_pick_orange.py",
    n_envs=1,
    trust_remote_code=True
)

# 访问环境
suite_name = next(iter(envs_dict))
sync_vector_env = envs_dict[suite_name][0]

# 从同步向量环境中获取 isaac 环境
env = sync_vector_env.envs[0].unwrapped

# 像使用普通 gym 环境一样使用它
obs, info = env.reset()

while True:
    # 随机采样动作
    action = torch.tensor(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### 运行

```bash
python envhub_random_action.py
```

运行后，你将看到 SO101 机械臂在纯随机命令下摆动。

---

## 遥操作

LeRobot 的遥操作栈可以驱动仿真机械臂。

### 步骤 1: 校准 Leader 设备

连接 SO101 Leader 控制器，运行以下校准命令：

```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader
```

### 步骤 2: 启动遥操作

```python
# envhub_teleop_example.py

import logging
import time
import gymnasium as gym

from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so_leader,
    bi_so_leader,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.envs import make_env


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    env_name: str = "so101_pick_orange"
    fps: int = 60


@dataclass
class EnvWrap:
    env: gym.Env


def make_env_from_leisaac(env_name: str = "so101_pick_orange"):
    """从 LeIsaac 创建环境"""
    envs_dict = make_env(
        f'LightwheelAI/leisaac_env:envs/{env_name}.py',
        n_envs=1,
        trust_remote_code=True
    )
    suite_name = next(iter(envs_dict))
    sync_vector_env = envs_dict[suite_name][0]
    env = sync_vector_env.envs[0].unwrapped
    
    return env


def teleop_loop(teleop: Teleoperator, env: gym.Env, fps: int):
    """遥操作主循环"""
    from leisaac.devices.action_process import preprocess_device_action
    from leisaac.assets.robots.lerobot import SO101_FOLLOWER_MOTOR_LIMITS
    from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

    env_wrap = EnvWrap(env=env)

    obs, info = env.reset()
    while True:
        loop_start = time.perf_counter()
        
        # 动态重置夹爪力限制
        if env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(env, 'so101leader')

        # 获取遥操作动作
        raw_action = teleop.get_action()
        
        # 处理动作
        processed_action = preprocess_device_action(
            dict(
                so101_leader=True,
                joint_state={
                    k.removesuffix(".pos"): v for k, v in raw_action.items()},
                motor_limits=SO101_FOLLOWER_MOTOR_LIMITS),
            env_wrap
        )
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(processed_action)
        
        if terminated or truncated:
            obs, info = env.reset()

        # 帧率控制
        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        
        loop_s = time.perf_counter() - loop_start
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1/loop_s:.0f} Hz)")


def teleoperate(cfg: TeleoperateConfig):
    """遥操作主函数"""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # 创建遥操作设备
    teleop = make_teleoperator_from_config(cfg.teleop)
    
    # 创建环境
    env = make_env_from_leisaac(cfg.env_name)

    # 连接设备
    teleop.connect()
    
    # 初始化环境
    if hasattr(env, 'initialize'):
        env.initialize()
    
    try:
        # 开始遥操作循环
        teleop_loop(teleop=teleop, env=env, fps=cfg.fps)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        env.close()


def main():
    teleoperate(TeleoperateConfig(
        teleop=so_leader.SO101LeaderConfig(
            port="/dev/ttyACM0",
            id='leader',
            use_degrees=False,
        ),
        env_name="so101_pick_orange",
        fps=60,
    ))


if __name__ == "__main__":
    main()
```

### 运行

```bash
python envhub_teleop_example.py
```

运行脚本后，你可以使用物理 Leader 设备操作仿真机械臂。

### 遥操作流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          遥操作数据流                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Leader 设备 (物理)                                                        │
│   ├── 获取电机位置/速度                                                      │
│   └── 发送原始动作                                                          │
│           ↓                                                                 │
│   preprocess_device_action()                                                │
│   ├── 转换为 Follower 格式                                                 │
│   ├── 应用电机限制                                                          │
│   └── 夹爪处理                                                              │
│           ↓                                                                 │
│   Isaac Lab 环境                                                            │
│   ├── 物理仿真                                                              │
│   ├── 渲染图像                                                              │
│   └── 返回观测/奖励                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 云端仿真 (无需 GPU)

没有本地 GPU 或正确的驱动程序？没问题！你可以在云端完全运行 LeIsaac，无需任何设置。

LeIsaac 在 **NVIDIA Brev** 上开箱即用，为你提供直接在浏览器中使用的完整配置环境。

### 开始使用

👉 **访问地址：** [https://lightwheelai.github.io/leisaac/docs/cloud_simulation/nvidia_brev](https://lightwheelai.github.io/leisaac/docs/cloud_simulation/nvidia_brev)

部署实例后，只需打开 **端口 80 (HTTP)** 的链接即可启动 **Visual Studio Code Server**（默认密码：`password`）。从那里，你可以运行仿真、编辑代码和可视化 IsaacLab 环境 —— 全部从你的网页浏览器完成。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        云端仿真优势                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✅ 无需 GPU                                                                │
│  ✅ 无需安装驱动程序                                                         │
│  ✅ 无需本地安装                                                            │
│  ✅ 只需点击即可运行                                                        │
│  ✅ 浏览器访问                                                              │
│  ✅ VS Code 集成                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 其他说明

我们保持 EnvHub 覆盖范围与 LeIsaac 任务对齐。当前支持的任务：

- `so101_pick_orange` - 拾取橙子
- `so101_lift_cube` - 抬立方体
- `so101_clean_toytable` - 清洁玩具桌
- `bi_so101_fold_cloth` - 双臂折叠布料

### 切换任务

通过在调用 `make_env` 时指定不同的脚本来切换任务：

```python
# 拾取橙子
envs_dict_pick_orange = make_env(
    "LightwheelAI/leisaac_env:envs/so101_pick_orange.py",
    n_envs=1,
    trust_remote_code=True
)

# 抬立方体
envs_dict_lift_cube = make_env(
    "LightwheelAI/leisaac_env:envs/so101_lift_cube.py",
    n_envs=1,
    trust_remote_code=True
)

# 清洁玩具桌
envs_dict_clean_toytable = make_env(
    "LightwheelAI/leisaac_env:envs/so101_clean_toytable.py",
    n_envs=1,
    trust_remote_code=True
)

# 折叠布料 (双臂)
envs_dict_fold_cloth = make_env(
    "LightwheelAI/leisaac_env:envs/bi_so101_fold_cloth.py",
    n_envs=1,
    trust_remote_code=True
)
```

### 特别注意：双臂任务

注意：在使用 `bi_so101_fold_cloth` 时，获取环境后需要立即调用 `initialize()`，然后再执行任何其他操作：

```python
import torch
from lerobot.envs import make_env

# 从 Hub 加载环境
envs_dict = make_env(
    "LightwheelAI/leisaac_env:envs/bi_so101_fold_cloth.py",
    n_envs=1,
    trust_remote_code=True
)

# 访问环境
suite_name = next(iter(envs_dict))
sync_vector_env = envs_dict[suite_name][0]
env = sync_vector_env.envs[0].unwrapped

# 注意：必须首先调用 initialize()
env.initialize()

# 其他环境操作...
```

---

## 总结

LeIsaac × LeRobot EnvHub 提供了一个强大的仿真到训练的工作流程：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        完整工作流程                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 启动仿真环境 ──→ Isaac Lab + LeIsaac                                    │
│          ↓                                                                  │
│  2. 遥操作数据收集 ──→ Leader 驱动 Follower                                  │
│          ↓                                                                  │
│  3. 数据转换 ──→ LeRobot Dataset 格式                                       │
│          ↓                                                                  │
│  4. 策略训练 ──→ ACT/Diffusion 等                                           │
│          ↓                                                                  │
│  5. 评估部署 ──→ 仿真或真实机器人                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 关键优势

- **一键启动**：一行代码加载环境
- **统一接口**：Gymnasium 风格
- **数据收集**：遥操作直接生成训练数据
- **云端支持**：无需 GPU，本地即可运行
- **持续更新**：LightWheel 团队维护

---

**参考资源：**
- [LeIsaac GitHub](https://github.com/LightwheelAI/leisaac)
- [Isaac Lab 文档](https://isaac-sim.github.io/)
- [NVIDIA Brev](https://brev.dev/)
- [LeRobot 官方文档](https://huggingface.co/docs/lerobot)

---

*翻译日期：2026-05-06*
