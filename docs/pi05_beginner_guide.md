# openpi 项目全景 & π₀.₅ 初学者认知地图

> 本文档系统梳理 Physical Intelligence `openpi` 开源仓库的核心架构、数据流与 π₀.₅ 的关键创新，帮助初学者快速建立整体认知。

---

## 目录

1. [项目定位：解决什么问题？](#一项目定位解决什么问题)
2. [π₀.₅ 核心创新速览](#二π₀₅-是什么先抓住三个核心创新)
3. [模型架构：从输入到输出的完整链路](#三模型架构从输入到输出的完整链路)
4. [训练流水线详解](#四训练流水线数据如何变成模型)
5. [推理部署方式](#五推理部署如何用在真实机器人上)
6. [π₀ vs π₀.₅ 差异速查](#六π₀-vs-π₀₅-核心差异速查表)
7. [初学者快速上手路径](#七初学者快速上手路径)
8. [关键认知锚点](#八给初学者的关键认知锚点)

---

## 一、项目定位：解决什么问题？

`openpi` 是 **Physical Intelligence** 开源的机器人大脑项目，实现 **Vision-Language-Action (VLA)** 模型。

核心目标：

> **让机器人通过"看"（视觉）+ "听指令"（语言）来生成"动作"（控制机械臂）**

输入：一张/多张摄像头画面 + 一句自然语言指令（如"把红色积木放到蓝色盘子里"）  
输出：机械臂未来一段时间内的关节动作序列

项目包含三个模型族：

| 模型 | 核心机制 | 特点 |
|------|---------|------|
| **π₀** | Flow Matching（流匹配） | 基线模型，动作生成看作"去噪"过程 |
| **π₀-FAST** | 自回归 + FAST 分词器 | 将动作离散化为 token，像生成文本一样生成动作 |
| **π₀.₅** | Flow Matching（改进版） | **更强的开放世界泛化能力**，支持"知识绝缘"训练 |

---

## 二、π₀.₅ 是什么？先抓住三个核心创新

π₀.₅ 是 π₀ 的升级版。改进不在于"模型更大"，而在于**输入表示和条件注入方式**的根本性改变：

### 🔑 创新 1：State 从"连续数字"变成"语言文本"

- **π₀ 的做法**：机器人关节状态通过 `state_proj` 线性层投影为向量，直接塞入 Action Expert 的输入流。
- **π₀.₅ 的做法**：**将 state 离散化成 256 个 bin，变成文本 token**，拼接到语言 prompt 中。

示例：
```
Task: pick up the cube, State: 128 45 200 ...;
Action:
```

**为什么这样做**：让预训练的 VLM 以"看文字"的方式理解机器人状态，更好地复用预训练知识。

### 🔑 创新 2：Timestep 条件注入升级为自适应归一化（adaRMSNorm）

- **π₀ 的做法**：Flow Matching 时间步 `t` 的 embedding 与 action token 拼接后通过 MLP 融合。
- **π₀.₅ 的做法**：`t` 经过独立时间 MLP 生成 `adarms_cond` 向量，注入到 Action Expert **每一层的 RMSNorm** 中，调制 scale、shift、gate。

**类比**：类似于 Diffusion 模型中的 AdaLN-Zero 或 DiT 中的自适应层归一化，让模型对"当前去噪进度"更敏感。

### 🔑 创新 3：知识绝缘（Knowledge Insulation）

π₀.₅ 在**训练方法**上的核心创新，解决以下问题：

> 微调时，机器人特定数据会"覆盖"掉预训练中学到的通用视觉-语言知识。

通过在训练过程中隔离/保护预训练知识，π₀.₅ 在没见过的新环境、新物体上表现更好。

---

## 三、模型架构：从输入到输出的完整链路

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           输入：Observation                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐  │
│  │  摄像头图像  │  │  机器人状态  │  │  语言指令 "pick up the cube"     │  │
│  │  (224×224)  │  │  (joint pos)│  │                                 │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────────────────┘  │
│         │                │                      │                        │
│         ▼                ▼                      ▼                        │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                     数据预处理流水线 (Transforms)                  │    │
│  │  1. Repack: 字段重命名                                            │    │
│  │  2. Robot Inputs: 机器人特定格式转换                                │    │
│  │  3. Normalize: Z-score 归一化 (需预计算 norm stats)                │    │
│  │  4. ResizeImages: 图像缩放到 224×224                               │    │
│  │  5. TokenizePrompt: 文本 → token IDs (π₀.₅ 会把 state 也 tokenize) │    │
│  │  6. PadStatesAndActions: 填充到固定维度                             │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                   │                                      │
│                                   ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                      Pi0 Model (Dual-Expert)                      │    │
│  │  ┌─────────────────────────┐    ┌─────────────────────────────┐  │    │
│  │  │   Expert 0: PaliGemma   │    │   Expert 1: Action Expert   │  │    │
│  │  │   (2B 参数，预训练 VLM)  │    │   (300M 参数，从头训练)      │  │    │
│  │  │                         │    │                             │  │    │
│  │  │   Prefix:               │◄──►│   Suffix:                   │  │    │
│  │  │   • 图像 patch tokens   │    │   • [state_token] (仅 π₀)   │  │    │
│  │  │   • 语言 prompt tokens  │    │   • action tokens (带噪声)   │  │    │
│  │  │                         │    │   • timestep embedding      │  │    │
│  │  │   Norm: 普通 RMSNorm    │    │   Norm: adaRMSNorm (仅 π₀.₅)│  │    │
│  │  └─────────────────────────┘    └─────────────────────────────┘  │    │
│  │  关键：两个专家共享同一个 Self-Attention，但 MLP 和 Norm 参数独立      │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                   │                                      │
│                                   ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                      Flow Matching 过程                           │    │
│  │                                                                     │    │
│  │   训练时：                                                           │    │
│  │   1. 从真实动作出发，随机采样噪声 noise 和时间 t                       │    │
│  │   2. 构造带噪动作：x_t = t·noise + (1-t)·actions                     │    │
│  │   3. 模型预测速度场 v_t，Loss = MSE(v_t, noise - actions)            │    │
│  │                                                                     │    │
│  │   推理时（Euler 积分）：                                              │    │
│  │   1. 从纯噪声 x_1 开始                                               │    │
│  │   2. 重复 num_steps 次：x_{t+dt} = x_t + dt · v_t                   │    │
│  │   3. 得到去噪后的动作 x_0                                            │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                   │                                      │
│                                   ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                     输出：Actions (动作块)                          │    │
│  │   shape = [action_horizon, action_dim]                            │    │
│  │   例如：未来 50 步，每步 14 个关节值（双臂 ALOHA）                    │    │
│  └──────────────────────────────────────────────────────────────────┘    │
```

### 关键组件说明

| 组件 | 文件位置 | 职责 |
|------|---------|------|
| **Base Model** | `src/openpi/models/model.py` | 定义 `Observation`、`Actions` 数据结构，抽象 `compute_loss()` / `sample_actions()` 接口 |
| **Pi0 Model** | `src/openpi/models/pi0.py` | 实现 Flow Matching 训练/推理逻辑，Dual-Expert 拼接前缀/后缀 |
| **Pi0 Config** | `src/openpi/models/pi0_config.py` | `Pi0Config` 通过 `pi05=True` 开关控制 π₀.₅ 特性 |
| **Gemma Backbone** | `src/openpi/models/gemma.py` | 多专家 Transformer，支持共享 Attention + 独立 MLP/Norm，含 `RMSNorm` / `adaRMSNorm` |
| **Tokenizer** | `src/openpi/models/tokenizer.py` | `PaligemmaTokenizer` 处理文本；π₀.₅ 模式下将 state 离散化为 bin token |
| **SigLIP ViT** | `src/openpi/models/siglip.py` | 将 224×224 图像编码为 patch token，无池化，直接送入语言模型 |

---

## 四、训练流水线：数据如何变成模型？

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  LeRobot 数据集  │────►│  数据转换流水线   │────►│   DataLoader    │
│  (HuggingFace)  │     │  (Transforms)    │     │  (Batch + Shard)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐             │
│   Checkpoint    │◄────│   Training Loop  │◄────────────┘
│   (orbax)       │     │   (JAX + FSDP)   │
└─────────────────┘     └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Model.compute_loss│
                    │  → Gradients      │
                    │  → Optimizer (AdamW)│
                    │  → EMA (可选)      │
                    └──────────────────┘
```

### 关键步骤详解

#### 1. 数据准备

- 数据集通常为 LeRobot 格式（HuggingFace 上的机器人数据集）
- 必须预计算 **norm stats**（归一化统计量）：
  ```bash
  uv run scripts/compute_norm_stats.py --config-name pi05_libero
  ```

#### 2. 配置系统

- 所有训练行为由 `TrainConfig`（`src/openpi/training/config.py`）定义
- 预设配置如 `pi05_libero`、`pi05_droid` 等
- π₀.₅ 配置关键：`pi05=True` → 启用 `discrete_state_input` + `adaRMSNorm`

#### 3. 训练启动

```bash
# 单卡微调
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_libero --exp-name=my_run --overwrite

# 多卡 FSDP
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_libero --exp-name=my_run --fsdp-devices 2
```

### 训练核心文件速查

| 文件 | 职责 |
|------|------|
| `scripts/train.py` | JAX 训练主入口：设备网格、训练循环、wandb 日志、checkpoint |
| `src/openpi/training/config.py` | 配置注册表 `_CONFIGS`，`TrainConfig` / `DataConfig` 定义 |
| `src/openpi/training/data_loader.py` | LeRobot / RLDS 数据加载，transform 应用，batch & shard |
| `src/openpi/training/checkpoints.py` | orbax 格式 checkpoint 存取 |
| `src/openpi/training/optimizer.py` | AdamW + Cosine LR schedule |
| `src/openpi/transforms.py` | 数据变换原子操作：Repack、Normalize、Tokenize、Resize 等 |

---

## 五、推理部署：如何用在真实机器人上？

### 模式 A：Policy Server（推荐用于真实机器人）

标准生产架构：**重模型跑在 GPU 服务器，轻客户端跑在机器人控制器**。

**Server 端（GPU 机器）：**
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=checkpoints/pi05_libero/my_run/20000
```

**Client 端（机器人控制器）：**
```python
from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy("ws://server_ip:8000")
action_chunk = policy.infer(observation)["actions"]
```

**关键约定**：
- 图像需在客户端 resize 到 **224×224**
- 图像格式需为 **uint8**
- state 无需客户端归一化（server 端处理）
- 通常每 **N 步**（replan interval）查询一次，开环执行 action chunk

### 模式 B：直接 Python 调用（适合调试 / 离线测试）

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# 1. 加载配置
config = _config.get_config("pi05_droid")

# 2. 下载/加载 checkpoint
checkpoint_dir = download.maybe_download(
    "gs://openpi-assets/checkpoints/pi05_droid"
)

# 3. 构建 Policy（自动加载归一化参数、transforms）
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 4. 推理
obs = {
    "image": camera_image,      # uint8, 224×224
    "state": robot_state,
    "prompt": "pick up the cube"
}
actions = policy.infer(obs)["actions"]  # [action_horizon, action_dim]
```

### 推理核心文件速查

| 文件 | 职责 |
|------|------|
| `src/openpi/policies/policy.py` | `Policy` 类：封装前处理 → `sample_actions` → 后处理 |
| `src/openpi/policies/policy_config.py` | `create_trained_policy()`：从 checkpoint 重建完整 Policy |
| `src/openpi/serving/websocket_policy_server.py` | WebSocket 策略服务器，默认监听 8000 端口 |
| `packages/openpi-client/` | 轻量客户端包，仅依赖 numpy、pillow、websockets |

---

## 六、π₀ vs π₀.₅ 核心差异速查表

| 维度 | π₀ | π₀.₅ |
|------|-----|------|
| **核心机制** | Flow Matching | Flow Matching |
| **State 输入** | 连续向量投影到 Action Expert | 离散化为语言 token，拼入 prompt |
| **Timestep 注入** | 与 action token 拼接后 MLP 融合 | adaRMSNorm，注入每层归一化 |
| **max_token_len** | 48 | 200（state 也占 token） |
| **训练方法** | 标准微调 | **知识绝缘**（保护预训练知识） |
| **泛化能力** | 标准 | **更强的开放世界/新任务泛化** |
| **支持的头** | Flow + FAST | **仅 Flow Matching** |
| **Checkpoint 前缀** | `pi0_*` | `pi05_*` |
| **Base Checkpoint** | `gs://openpi-assets/checkpoints/pi0_base` | `gs://openpi-assets/checkpoints/pi05_base` |

---

## 七、初学者快速上手路径

### Step 0：环境准备

```bash
# 1. 克隆（带子模块）
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# 2. 安装依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Step 1：跑通无机器人推理测试

```bash
# 终端 1：启动策略服务（使用预训练模型）
uv run scripts/serve_policy.py --env=ALOHA_SIM

# 终端 2：运行简单客户端（发送随机观测，测试端到端链路）
uv run examples/simple_client/main.py --env ALOHA_SIM
```

### Step 2：理解数据流

重点阅读 `src/openpi/transforms.py` 中的关键 transform：

| Transform | 作用 |
|-----------|------|
| `Normalize` / `Unnormalize` | Z-score 或分位数归一化 |
| `TokenizePrompt` | 文本 → token IDs（π₀.₅ 含 state tokenization） |
| `ResizeImages` | 图像 resize + pad 到 224×224 |
| `DeltaActions` / `AbsoluteActions` | 绝对动作与增量动作互转 |

### Step 3：跑 LIBERO 仿真基准

```bash
# 使用 Docker Compose 一键跑 LIBERO 评估
sudo xhost +local:docker
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

### Step 4：尝试微调（需准备自己的数据）

```bash
# 1. 将数据转换为 LeRobot 格式
#    参考：examples/libero/convert_libero_data_to_lerobot.py

# 2. 计算归一化统计量
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 3. 启动训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_libero --exp-name=my_first_run --overwrite

# 4. 部署微调后的模型
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=checkpoints/pi05_libero/my_first_run/20000
```

---

## 八、给初学者的关键认知锚点

1. **"动作是一块（chunk），不是一步"**  
   模型一次预测未来 `action_horizon` 步的动作（如 50 步）。机器人可以开环执行其中前几步，然后重新查询模型，这叫做 **action chunking**。

2. **"训练时加噪，推理时去噪"**  
   Flow Matching 的训练和推理是对称的：训练时学"从带噪动作预测真实方向"，推理时沿着这个方向一步步从噪声走回真实动作。

3. **"Transforms 是训练和推理的契约"**  
   `Policy` 类确保训练和推理使用**完全相同的 transforms**。这是整个项目数据一致性的核心保障。如果训练时做了归一化而推理时忘了反归一化，机器人会收到乱码指令。

4. **"π₀.₅ 的本质是更好地利用预训练 VLM"**  
   它的所有改动（state → text、adaRMSNorm、知识绝缘）都围绕一个目标：**不要让微调数据破坏掉 PaliGemma 预训练中学到的通用世界知识**。

5. **"Dual-Expert 不是两个独立模型"**  
   PaliGemma Expert（2B）和 Action Expert（300M）**共享 Self-Attention**，但拥有独立的 MLP 和 Norm 参数。这让视觉/语言信息能直接影响动作生成，同时保持动作流的独立优化空间。

---

*文档生成时间：2026-04-28*  
*基于 openpi 仓库代码分析整理*
