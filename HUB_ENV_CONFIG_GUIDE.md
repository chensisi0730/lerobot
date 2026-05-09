# HubEnvConfig 详细说明

> 从 HuggingFace Hub 动态加载和执行自定义环境

## 目录

- [什么是 HubEnvConfig](#什么是-hubenvconfig)
- [工作原理](#工作原理)
- [使用方法](#使用方法)
- [创建 Hub 环境](#创建-hub-环境)
- [安全机制](#安全机制)
- [完整示例](#完整示例)
- [常见问题](#常见问题)

---

## 什么是 HubEnvConfig

`HubEnvConfig` 是 LeRobot 提供的一种特殊环境配置类，允许你从 **HuggingFace Hub** 动态加载自定义环境，而无需在本地安装任何代码。

### 核心优势

```
┌─────────────────────────────────────────────────────────────────────────┐
│  传统方式:                                                               │
│  1. 编写环境代码                                                         │
│  2. 打包成 Python 包                                                    │
│  3. 发布到 PyPI                                                         │
│  4. 用户 pip install                                                    │
│  5. 才能使用环境                                                        │
│                                                                         │
│  HubEnvConfig 方式:                                                      │
│  1. 编写环境代码 (env.py)                                                │
│  2. 上传到 HuggingFace Hub                                              │
│  3. 用户直接使用 (无需安装)                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 适用场景

- **快速原型开发**：快速迭代环境代码，无需重新发布
- **共享环境**：与他人分享自定义环境
- **研究复现**：提供环境的完整实现，确保可复现性
- **自定义机器人**：为特定机器人创建专用环境

---

## 工作原理

### 架构流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HubEnvConfig 工作流程                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  步骤1: 配置 HubEnvConfig                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  hub_path = "username/my-env@main:env.py"                           │  │
│  │  ├── username/my-env: HuggingFace Hub 仓库 ID                       │  │
│  │  ├── @main: Git 分支/标签/提交 (可选)                                │  │
│  │  └── :env.py: 环境文件路径 (默认 env.py)                             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                          ↓                                                  │
│  步骤2: 下载环境文件                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  hf_hub_download()                                                   │  │
│  │  ├── 从 Hub 下载 env.py 到本地缓存                                   │  │
│  │  └── 支持版本锁定 (通过 @revision)                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                          ↓                                                  │
│  步骤3: 动态导入模块                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  _import_hub_module()                                                │  │
│  │  ├── 使用 importlib 动态加载 Python 文件                             │  │
│  │  └── 检查是否包含 make_env() 函数                                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                          ↓                                                  │
│  步骤4: 调用 make_env()                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  module.make_env(n_envs=4, use_async_envs=True)                     │  │
│  │  ├── 创建环境实例                                                    │  │
│  │  └── 返回 VectorEnv 或 Dict 格式                                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                          ↓                                                  │
│  步骤5: 标准化返回结果                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  _normalize_hub_result()                                             │  │
│  │  ├── 统一返回格式: {suite: {task_id: vec_env}}                      │  │
│  │  └── 支持单环境、VectorEnv、Dict 多种格式                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. HubEnvConfig 类

```python
@dataclass
class HubEnvConfig(EnvConfig):
    """从 HuggingFace Hub 加载环境的配置类"""
    
    hub_path: str | None = None  # Hub 仓库路径
    
    @property
    def gym_kwargs(self) -> dict:
        # Hub 环境不需要 gym_kwargs，由 make_env() 处理
        return {}
```

#### 2. make_env() 工厂函数

```python
def make_env(
    cfg: EnvConfig | str,
    n_envs: int = 1,
    use_async_envs: bool = False,
    hub_cache_dir: str | None = None,
    trust_remote_code: bool = False,  # 安全开关
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """
    创建环境实例
    
    参数:
        cfg: 环境配置或 Hub 路径字符串
        n_envs: 并行环境数量
        use_async_envs: 是否使用异步环境
        hub_cache_dir: Hub 文件缓存目录
        trust_remote_code: 是否信任远程代码 (必须为 True)
    
    返回:
        {suite_name: {task_id: VectorEnv}}
    """
```

#### 3. 辅助函数

```python
# 解析 Hub URL
_parse_hub_url("username/repo@main:env.py")
# → ("username/repo", "main", "env.py")

# 下载 Hub 文件
_download_hub_file(hub_path, trust_remote_code=True, hub_cache_dir=None)
# → (repo_id, file_path, local_file, revision)

# 导入模块
_import_hub_module(local_file, repo_id)
# → Python 模块对象

# 调用 make_env
_call_make_env(module, n_envs=4, use_async_envs=False, cfg=None)
# → 环境实例

# 标准化结果
_normalize_hub_result(raw_result)
# → {suite: {task_id: vec_env}}
```

---

## 使用方法

### 方式1: 使用字符串路径 (推荐)

```python
from lerobot.envs import make_env

# 直接传入 Hub 路径字符串
envs = make_env(
    "username/my-robot-env@v1.0:env.py",
    n_envs=4,
    trust_remote_code=True,  # 必须设置为 True
)

# 使用环境
obs, info = envs.reset()
action = policy.select_action(obs)
obs, reward, done, truncated, info = envs.step(action)
```

### 方式2: 使用 HubEnvConfig 配置类

```python
from lerobot.envs import HubEnvConfig, make_env

# 创建配置
env_cfg = HubEnvConfig(
    hub_path="username/my-robot-env@main:env.py",
    fps=30,
    features={
        "observation.image": {"shape": [3, 480, 640], "dtype": "video"},
        "observation.state": {"shape": [7], "dtype": "float32"},
        "action": {"shape": [7], "dtype": "float32"},
    },
)

# 创建环境
envs = make_env(
    env_cfg,
    n_envs=4,
    use_async_envs=True,
    trust_remote_code=True,
)
```

### 方式3: 在训练配置中使用

```python
from lerobot.configs.train import TrainPipelineConfig
from lerobot.envs import HubEnvConfig

config = TrainPipelineConfig(
    dataset=DatasetConfig(repo_id="my_dataset"),
    env=HubEnvConfig(
        hub_path="username/my-robot-env@v1.0:env.py",
        fps=30,
    ),
    policy=ACTConfig(),
)

# 训练时会自动从 Hub 加载环境进行评估
from lerobot.scripts.lerobot_train import train
train(config)
```

### Hub 路径格式说明

```
格式: "repo_id[@revision][:file_path]"

示例:
├── "username/my-env"                    # 默认 main 分支，env.py 文件
├── "username/my-env@v1.0"               # 指定标签 v1.0
├── "username/my-env@dev"                # 指定分支 dev
├── "username/my-env@abc123"             # 指定提交 hash
├── "username/my-env:custom_env.py"      # 指定文件路径
└── "username/my-env@v1.0:envs/main.py"  # 完整指定
```

---

## 创建 Hub 环境

### 1. 编写 env.py 文件

创建一个包含 `make_env()` 函数的 Python 文件：

```python
# env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyCustomEnv(gym.Env):
    """自定义环境"""
    
    def __init__(self):
        super().__init__()
        
        # 定义观测空间
        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
        })
        
        # 定义动作空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        
        # 初始化环境状态
        self.state = None
        self.step_count = 0
        self.max_steps = 500
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(7, dtype=np.float32)
        self.step_count = 0
        
        obs = {
            "pixels": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "agent_pos": self.state,
        }
        info = {"task_description": "My custom task"}
        
        return obs, info
    
    def step(self, action):
        # 更新状态
        self.state = self.state + action
        self.step_count += 1
        
        # 计算奖励
        reward = -np.linalg.norm(self.state)
        
        # 判断是否结束
        done = self.step_count >= self.max_steps
        truncated = False
        
        # 生成观测
        obs = {
            "pixels": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "agent_pos": self.state,
        }
        info = {"task_description": "My custom task"}
        
        return obs, reward, done, truncated, info
    
    def render(self):
        # 可选：渲染环境
        pass


def make_env(n_envs: int = 1, use_async_envs: bool = False, cfg=None):
    """
    创建环境的入口函数
    
    参数:
        n_envs: 并行环境数量
        use_async_envs: 是否使用异步环境
        cfg: 可选的 HubEnvConfig 配置对象
    
    返回:
        VectorEnv 或 Dict 格式的环境
    """
    if n_envs == 1:
        # 单环境
        return MyCustomEnv()
    else:
        # 多环境
        if use_async_envs:
            return gym.vector.AsyncVectorEnv(
                [MyCustomEnv for _ in range(n_envs)]
            )
        else:
            return gym.vector.SyncVectorEnv(
                [MyCustomEnv for _ in range(n_envs)]
            )
```

### 2. 创建 HuggingFace Hub 仓库

```bash
# 创建仓库结构
my-robot-env/
├── env.py              # 环境代码 (必需)
├── README.md           # 说明文档 (推荐)
├── requirements.txt    # 依赖列表 (可选)
└── assets/             # 资源文件 (可选)
    └── preview.png
```

### 3. 上传到 HuggingFace Hub

```bash
# 登录 HuggingFace
huggingface-cli login

# 创建仓库
huggingface-cli repo create my-robot-env --type dataset

# 上传文件
cd my-robot-env
git init
git add .
git commit -m "Initial commit"
git remote add origin https://huggingface.co/datasets/username/my-robot-env
git push -u origin main
```

### 4. 添加版本标签

```bash
# 创建版本标签
git tag v1.0
git push origin v1.0
```

---

## 安全机制

### trust_remote_code 参数

HubEnvConfig 执行来自 HuggingFace Hub 的任意 Python 代码，存在安全风险。因此，**必须显式设置 `trust_remote_code=True`**：

```python
# ❌ 错误：未设置 trust_remote_code
envs = make_env("username/my-env")  # 会抛出 RuntimeError

# ✅ 正确：显式信任远程代码
envs = make_env("username/my-env", trust_remote_code=True)
```

### 安全建议

```
┌─────────────────────────────────────────────────────────────────────────┐
│  安全最佳实践:                                                           │
│                                                                         │
│  1. 只使用可信来源的环境                                                 │
│     ├── 官方 LeRobot 环境                                               │
│     ├── 知名研究机构发布的环境                                           │
│     └── 你自己或团队发布的环境                                           │
│                                                                         │
│  2. 锁定版本号                                                           │
│     ├── 使用 @v1.0 标签而非 @main                                       │
│     ├── 或使用完整的 commit hash                                        │
│     └── 避免使用可变分支 (@main, @dev)                                  │
│                                                                         │
│  3. 审查代码                                                             │
│     ├── 在 Hub 上查看 env.py 源代码                                     │
│     ├── 检查是否有可疑操作                                              │
│     └── 查看 requirements.txt 依赖                                     │
│                                                                         │
│  4. 使用沙箱环境                                                         │
│     ├── 在 Docker 容器中运行                                            │
│     └── 限制网络和文件系统访问                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 错误示例

```python
# 安全错误示例
RuntimeError: Refusing to execute remote code from the Hub for 'username/my-env'. 
Executing hub env modules runs arbitrary Python code from third-party repositories. 
If you trust this repo and understand the risks, call `make_env(..., trust_remote_code=True)` 
and prefer pinning to a specific revision: 'user/repo@<commit-hash>:env.py'.
```

---

## 完整示例

### 示例1: 简单的 Hub 环境

**env.py:**

```python
import gymnasium as gym
import numpy as np

class SimpleEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 1, (4,))
        self.action_space = gym.spaces.Box(-1, 1, (2,))
        self.state = None
    
    def reset(self, seed=None):
        self.state = np.random.rand(4)
        return self.state, {}
    
    def step(self, action):
        self.state = self.state + action[:2] / 10
        reward = -np.sum(self.state**2)
        return self.state, reward, False, False, {}

def make_env(n_envs=1, use_async_envs=False, cfg=None):
    if n_envs == 1:
        return SimpleEnv()
    return gym.vector.SyncVectorEnv([SimpleEnv for _ in range(n_envs)])
```

**使用:**

```python
from lerobot.envs import make_env

envs = make_env(
    "username/simple-env@v1.0:env.py",
    n_envs=4,
    trust_remote_code=True,
)

obs, info = envs.reset()
print(f"Observation shape: {obs['agent_pos'].shape}")
```

### 示例2: 多任务 Hub 环境

**env.py:**

```python
import gymnasium as gym
from typing import Dict

class TaskA(gym.Env):
    # 任务 A 实现
    pass

class TaskB(gym.Env):
    # 任务 B 实现
    pass

def make_env(n_envs=1, use_async_envs=False, cfg=None) -> Dict:
    """
    返回多任务环境字典
    
    返回格式: {suite_name: {task_id: VectorEnv}}
    """
    return {
        "my_suite": {
            0: gym.vector.SyncVectorEnv([TaskA for _ in range(n_envs)]),
            1: gym.vector.SyncVectorEnv([TaskB for _ in range(n_envs)]),
        }
    }
```

**使用:**

```python
from lerobot.envs import make_env

envs = make_env(
    "username/multi-task-env:env.py",
    n_envs=2,
    trust_remote_code=True,
)

# envs = {"my_suite": {0: VectorEnv, 1: VectorEnv}}
for suite_name, tasks in envs.items():
    for task_id, vec_env in tasks.items():
        print(f"Suite: {suite_name}, Task: {task_id}")
        obs, info = vec_env.reset()
```

### 示例3: 使用 HubEnvConfig 配置

```python
from lerobot.envs import HubEnvConfig, make_env
from lerobot.configs import PolicyFeature, FeatureType

# 创建详细配置
env_cfg = HubEnvConfig(
    hub_path="username/my-robot-env@v1.0:env.py",
    fps=30,
    features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        "observation.image": PolicyFeature(
            type=FeatureType.VISUAL, 
            shape=(3, 480, 640)
        ),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    },
    features_map={
        "action": "action",
        "pixels": "observation.image",
        "agent_pos": "observation.state",
    },
)

# 创建环境
envs = make_env(
    env_cfg,
    n_envs=4,
    use_async_envs=True,
    trust_remote_code=True,
)
```

---

## 常见问题

### Q1: 如何调试 Hub 环境？

```python
# 方法1: 本地测试
python env.py  # 直接运行 env.py 测试

# 方法2: 使用本地文件
from lerobot.envs.utils import _load_module_from_path

module = _load_module_from_path("./env.py")
env = module.make_env(n_envs=1)

# 方法3: 查看下载的文件
# Hub 文件会缓存到 ~/.cache/huggingface/hub/
```

### Q2: 如何处理依赖问题？

**requirements.txt:**

```
gymnasium>=0.29.0
numpy>=1.24.0
# 其他依赖...
```

**安装依赖:**

```bash
# 用户需要手动安装依赖
pip install -r https://huggingface.co/datasets/username/my-env/raw/main/requirements.txt
```

### Q3: 如何支持 Isaac Lab 环境？

```python
# env.py
from isaaclab.envs import ManagerBasedRLEnv

def make_env(n_envs=1, use_async_envs=False, cfg=None):
    from isaaclab_tasks.manager_based.manipulation.reach.config.franka_panda_ik_env import (
        cfg_franka_panda_ik_env,
    )
    
    env_cfg = cfg_franka_panda_ik_env.EnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    if n_envs == 1:
        return env
    else:
        return gym.vector.SyncVectorEnv([lambda: env for _ in range(n_envs)])
```

### Q4: 返回格式有什么要求？

```python
# 支持的返回格式:

# 1. 单环境
def make_env(n_envs=1, use_async_envs=False, cfg=None):
    return MyEnv()  # gym.Env

# 2. VectorEnv
def make_env(n_envs=4, use_async_envs=False, cfg=None):
    return gym.vector.SyncVectorEnv([MyEnv for _ in range(4)])

# 3. 多任务字典
def make_env(n_envs=1, use_async_envs=False, cfg=None):
    return {
        "suite_name": {
            0: gym.vector.SyncVectorEnv([TaskA]),
            1: gym.vector.SyncVectorEnv([TaskB]),
        }
    }
```

### Q5: 如何更新 Hub 环境？

```bash
# 1. 修改 env.py
vim env.py

# 2. 提交更改
git add env.py
git commit -m "Fix bug in environment"
git push

# 3. 创建新版本标签
git tag v1.1
git push origin v1.1

# 4. 用户使用新版本
envs = make_env("username/my-env@v1.1:env.py", trust_remote_code=True)
```

---

## 总结

HubEnvConfig 提供了一种灵活、便捷的方式来共享和使用自定义环境：

**优点:**
- ✅ 无需安装，即开即用
- ✅ 支持版本控制
- ✅ 易于共享和复现
- ✅ 支持多任务环境

**注意事项:**
- ⚠️ 必须显式信任远程代码
- ⚠️ 只使用可信来源的环境
- ⚠️ 建议锁定版本号
- ⚠️ 用户需手动安装依赖

**最佳实践:**
- 📝 提供详细的 README 文档
- 🏷️ 使用语义化版本标签
- 📦 明确列出依赖项
- 🔒 定期审查代码安全性

---

**参考资源:**
- [HuggingFace Hub 文档](https://huggingface.co/docs/hub)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [LeRobot 环境示例](https://huggingface.co/lerobot)
