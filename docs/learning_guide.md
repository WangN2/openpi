# openpi 项目系统性学习指南

## 前言

本文档目标是帮助你从零开始，系统性地掌握 [Physical Intelligence 的 openpi](https://github.com/Physical-Intelligence/openpi) 项目，最终能够熟练完成 pi0.5 的训练、评估、推理和迁移（fine-tuning）。

---

## 第一部分：项目概览

### 1.1 这是什么？

openpi 是一个**视觉-语言-动作（VLA）模型**的开源项目。它可以让机器人通过摄像头看到场景、理解语言指令，然后输出动作。项目提供三种模型：

| 模型 | 核心思路 | 特点 |
|------|---------|------|
| **pi0 (π₀)** | Flow-based VLA | 扩散/流匹配方式生成动作，连续动作空间 |
| **pi0-FAST (π₀-FAST)** | Autoregressive VLA | 自回归方式生成离散动作 token，速度更快 |
| **pi0.5 (π₀.₅)** | 升级版 pi0 | 更好的开放世界泛化能力，引入 knowledge insulation |

**你的目标是 pi0.5**，它在 pi0 的基础上做了两项关键改进：
1. **离散状态输入（discrete state input）**：机器人的状态（关节角度等）不再作为连续特征拼接到 action expert，而是通过离散化后作为文本 token 拼接到 prompt 中
2. **adaRMSNorm**：在 action expert 中使用自适应 RMSNorm，将 flow matching 的时间步信息注入到特征变换中

### 1.2 技术栈

| 层 | 技术 |
|---|------|
| Python | >= 3.11 |
| 包管理 | uv |
| JAX 框架 | JAX 0.5.3 + Flax NNX |
| PyTorch 框架 | PyTorch 2.7.1 |
| 视觉编码 | SigLIP (So400m/14) |
| 语言模型 | PaliGemma (2B 参数) |
| 动作专家 | 小型 Gemma (300M 参数) |
| 分词器 | SentencePiece (PaliGemma tokenizer) |
| 数据集 | LeRobot / RLDS (DROID) |
| 检查点 | Orbax (JAX) / safetensors (PyTorch) |
| 日志 | wandb |
| 代码质量 | ruff (格式化+linter) |

### 1.3 项目结构

```
openpi/
├── src/openpi/
│   ├── models/              # JAX 模型实现（核心算法）
│   │   ├── model.py         # 基类：BaseModelConfig, BaseModel, Observation, Actions
│   │   ├── pi0.py           # pi0 和 pi0.5 的流匹配模型
│   │   ├── pi0_fast.py      # pi0-FAST 自回归模型
│   │   ├── pi0_config.py    # Pi0Config 配置类
│   │   ├── gemma.py         # PaliGemma 语言模型实现
│   │   ├── siglip.py        # SigLIP 视觉编码器
│   │   ├── lora.py          # LoRA 低秩适配
│   │   └── tokenizer.py     # 分词器
│   ├── models_pytorch/      # PyTorch 模型副本
│   │   ├── pi0_pytorch.py   # PyTorch 版 pi0/pi0.5
│   │   └── gemma_pytorch.py # PyTorch 版 PaliGemma
│   ├── policies/            # 策略封装（推理入口）
│   │   ├── policy.py        # Policy 类：transform + model infer
│   │   ├── policy_config.py # create_trained_policy()
│   │   ├── aloha_policy.py  # ALOHA 机器人输入/输出转换
│   │   ├── droid_policy.py  # DROID 机器人输入/输出转换
│   │   └── libero_policy.py # LIBERO 环境输入/输出转换
│   ├── training/
│   │   ├── config.py        # 中央配置注册表（所有 TrainConfig）
│   │   ├── data_loader.py   # 统一数据加载器
│   │   ├── checkpoints.py   # 检查点保存/恢复
│   │   ├── optimizer.py     # AdamW + 学习率调度
│   │   ├── weight_loaders.py# 权重加载器
│   │   ├── sharding.py      # JAX FSDP 配置
│   │   └── utils.py         # TrainState 等工具
│   ├── serving/
│   │   └── websocket_policy_server.py  # WebSocket 推理服务
│   ├── transforms.py        # 数据变换管线
│   └── shared/              # 共享工具函数
├── packages/openpi-client/  # 轻量级客户端（WebSocket 通信）
├── scripts/
│   ├── train.py             # JAX 训练入口
│   ├── train_pytorch.py     # PyTorch 训练入口
│   ├── serve_policy.py      # 策略服务入口
│   └── compute_norm_stats.py# 计算归一化统计量
└── examples/                # 各机器人平台的使用示例
```

---

## 第二部分：核心概念

### 2.1 VLA 模型架构（以 pi0.5 为例）

pi0.5 是一个**流匹配（Flow Matching）**模型。它的核心结构如下：

```
                     ┌─────────────────────────────┐
                     │        输出动作 (32-dim)      │
                     └─────────────┬───────────────┘
                                   │
                     ┌─────────────▼───────────────┐
                     │      action_out_proj         │  Linear(300->32)
                     │      预测速度场 v_t          │
                     └─────────────┬───────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │ pi0.5:                 │ pi0:                    │
          │ time_mlp_in/out        │ action_time_mlp_in/out │
          │ (adaRMS 条件)          │ (拼接 time+action)     │
          └─────────────┬──────────┴───────────┬────────────┘
                        │                      │
          ┌─────────────▼──────────┐ ┌─────────▼──────────┐
          │   Action Expert        │ │   State Proj       │  (仅 pi0)
          │   (Gemma 300M)         │ │                    │
          │   处理动作 token + time │ │   state → token    │
          └─────────────┬──────────┘ └────────────────────┘
                        │
          ┌─────────────▼──────────────────────────────────┐
          │         PaliGemma (Gemma 2B) 主干              │
          │   [图像token | 语言token | 动作token]           │
          │    Prefix: 双向注意力                           │
          │    Suffix: 因果注意力 + KV Cache                │
          └─────────────┬──────────────────────────────────┘
                        │
          ┌─────────────▼──────────┐  ┌───────────────────┐
          │   SigLIP 视觉编码器    │  │   Prompt Tokenizer │
          │   (So400m/14)         │  │  + 离散化状态       │
          │   图像 → visual tokens │  │  文本 → text tokens│
          └────────────────────────┘  └───────────────────┘
```

**pi0 vs pi0.5 的关键区别**：

| 特性 | pi0 | pi0.5 |
|------|-----|-------|
| 状态输入方式 | 连续向量拼接在 suffix | 离散化为文本 token 拼入 prefix |
| 时间步注入 | 与 action token 拼接 | 通过 adaRMSNorm 条件注入 |
| max_token_len | 48 | 200 |
| prompt 格式 | `"{task}\n"` | `"Task: {task}, State: {s0 s1 ...};\nAction: "` |
| 归一化方式 | z-score | quantile |

### 2.2 流匹配（Flow Matching）原理

pi0/pi0.5 使用**流匹配**来生成动作，不是扩散（但二者很相似）：

1. **训练时**：
   - 从数据中取一个真实动作 `actions` 
   - 采样高斯噪声 `noise`
   - 采样时间步 `t ~ Beta(1.5, 1)`
   - 构造插值：`x_t = t * noise + (1-t) * actions`
   - 目标速度场：`u_t = noise - actions`
   - 模型预测 `v_t`，损失 = MSE(v_t, u_t)

2. **推理时**（反向 ODE 求解）：
   - 从纯噪声 `x_1 = noise` 开始
   - 用欧拉法逐步去噪：`x_{t-dt} = x_t + v_t * dt`
   - `num_steps` 步后得到 `x_0`（预测动作）
   - 默认 `num_steps=10`

### 2.3 数据管线（Transforms）

数据从原始数据集到模型输入，经过**三阶段变换**：

```
原始数据集格式 (LeRobot/RLDS)
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: repack_transforms │  重映射键名，使不同数据集格式统一
│  RepackTransform            │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 2: data_transforms   │  机器人特定变换
│  DroidInputs / AlohaInputs  │  如：关节→delta、笛卡尔→joint
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Normalize                  │  归一化 state 和 actions
│  (z-score 或 quantile)      │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 3: model_transforms  │  模型特定变换
│  TokenizePrompt             │  分词、离散化状态、resize 图像
│  ResizeImages(224,224)      │
│  PadStatesAndActions        │
└─────────────────────────────┘
        │
        ▼
       Observation/模型输入
```

### 2.4 模型输入格式（Observation）

这是 `model.py` 中定义的核心数据结构：

```python
Observation(
    images={
        "base_0_rgb": Float[*b, 224, 224, 3],         # [-1, 1]
        "left_wrist_0_rgb": Float[*b, 224, 224, 3],
        "right_wrist_0_rgb": Float[*b, 224, 224, 3],
    },
    image_masks={
        "base_0_rgb": Bool[*b],    # True=有效
        ...
    },
    state: Float[*b, action_dim],   # 机器人状态
    tokenized_prompt: Int32[*b, max_token_len],    # 分词后的 prompt
    tokenized_prompt_mask: Bool[*b, max_token_len], # attention mask
)
Actions: Float[*b, action_horizon, action_dim]  # 动作序列
```

---

## 第三部分：配置系统（最重要）

所有训练和推理行为都由 **TrainConfig** 控制，集中注册在 [config.py](../src/openpi/training/config.py) 中。

### 3.1 TrainConfig 关键字段

```python
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: str                        # 配置名称（唯一标识）
    model: BaseModelConfig           # 模型配置
    weight_loader: WeightLoader      # 预训练权重加载
    data: DataConfigFactory          # 数据配置
    batch_size: int = 32
    num_train_steps: int = 30_000
    lr_schedule: LRScheduleConfig    # 学习率调度
    optimizer: OptimizerConfig       # 优化器
    ema_decay: float | None = 0.99  # EMA 衰减（LoRA 时关闭）
    freeze_filter: Filter            # 冻结哪些参数（LoRA）
    fsdp_devices: int = 1            # FSDP 设备数（>1 启用 FSDP）
    pytorch_weight_path: str | None  # PyTorch 权重路径
```

### 3.2 Pi0Config 关键字段

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(BaseModelConfig):
    action_dim: int = 32          # 动作空间维度（pi0.5 用32）
    action_horizon: int = 50      # 动作预测长度
    max_token_len: int = 200      # pi0.5 用200（pi0 用48）
    pi05: bool = True             # 是否启用 pi0.5 模式
    discrete_state_input: bool = True  # 离散化状态输入
    paligemma_variant: str = "gemma_2b"       # 视觉语言模型
    action_expert_variant: str = "gemma_300m" # 动作专家模型
    dtype: str = "bfloat16"
```

### 3.3 预置配置

`_CONFIGS` 列表中包含 20+ 个预置配置，按用途分类：

| 配置名 | 用途 |
|-------|------|
| `pi0_aloha` | ALOHA 推理（pi0） |
| `pi05_aloha` | ALOHA 推理（pi0.5） |
| `pi0_droid` | DROID 推理（pi0） |
| `pi05_droid` | DROID 推理（pi0.5） |
| `pi05_libero` | LIBERO 推理（pi0.5） |
| `pi0_aloha_sim` | ALOHA sim 训练 |
| `pi0_libero` | LIBERO 微调训练（全量） |
| `pi0_libero_low_mem_finetune` | LIBERO 微调训练（LoRA） |
| `pi05_libero` | pi0.5 LIBERO 微调 |
| `pi0_fast_full_droid_finetune` | pi0-FAST DROID 全量微调 |
| `pi05_full_droid_finetune` | pi0.5 DROID 全量微调 |
| `debug` | 快速调试（dummy 模型） |
| `debug_pi05` | pi0.5 快速调试 |

---

## 第四部分：训练流程详解

### 4.1 完整训练步骤

```
1. 准备数据集 ─────────────────────────────┐
   收集数据 → 转换为 LeRobot 格式           │
                                           │
2. 计算归一化统计量 ────────────────────────┤
   uv run scripts/compute_norm_stats.py     │
   --config-name <你的配置>                 │
                                           │
3. 配置训练参数 ────────────────────────────┤
   在 config.py 中注册新的 TrainConfig      │
   或使用现有配置                           │
                                           │
4. [可选] 准备 PyTorch 环境 ────────────────┤
   如果要使用 PyTorch 训练                  │
   需要 apply transformers_replace 补丁     │
                                           │
5. 启动训练 ───────────────────────────────┤
   JAX:    uv run scripts/train.py <配置>   │
   PyTorch: torchrun ... train_pytorch.py  │
                                           │
6. 监控训练 ───────────────────────────────┤
   wandb 查看 loss / grad_norm             │
   检查点自动保存                           │
                                           │
7. 推理部署 ───────────────────────────────┤
   uv run scripts/serve_policy.py           │
   --policy.config=<配置>                   │
   --policy.dir=<检查点目录>               │
```

### 4.2 JAX 训练（train.py）

核心训练循环在 `scripts/train.py` 中：

```
train.py 主要流程
─────────────────
1. 初始化日志、wandb
2. 创建数据加载器 (create_data_loader)
3. 初始化模型 (config.model.create)
   → 加载预训练权重 (weight_loader)
4. 创建 TrainState (参数 + 优化器状态 + EMA)
5. 编译训练步 (jax.jit(train_step))
6. 循环:
   a. 取 batch
   b. train_step: 前向 → 计算 loss → 梯度 → 更新参数 → EMA
   c. 日志记录
   d. 定期保存检查点
```

**训练步详解**（[train.py:137](../scripts/train.py#L137)）：

```python
def train_step(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    
    def loss_fn(model, rng, observation, actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)
    
    # 计算梯度（只更新可训练参数）
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(...)
    updates, new_opt_state = tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    # EMA 更新
    if ema_decay is not None:
        ema_params = decay * old + (1 - decay) * new
    
    return new_state, info
```

### 4.3 PyTorch 训练（train_pytorch.py）

PyTorch 训练支持 DDP（分布式数据并行），适合多 GPU 场景：

```bash
# 单卡
uv run scripts/train_pytorch.py <配置> --exp_name <名称>

# 多卡（单节点）
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py <配置> --exp_name <名称>
```

需要注意的是：PyTorch 路径目前**不完全支持** pi0-FAST、FSDP、LoRA、EMA。

### 4.4 数据加载器

支持两种数据源：

**LeRobot 格式**（推荐，适合 <100 小时数据）：
```python
dataset = LeRobotDataset(repo_id)
# 需要提供 delta_timestamps 来生成动作序列
```

**RLDS 格式**（适合大规模 DROID 数据集）：
```python
dataset = DroidRldsDataset(data_dir, batch_size, ...)
```

数据变换管线应用顺序：
```python
transforms = [
    *repack_transforms,    # 重映射键名
    *data_transforms,      # 机器人特定变换
    Normalize(norm_stats), # 归一化
    *model_transforms,     # 模型特定变换（分词、resize、padding）
]
```

---

## 第五部分：模型推理与部署

### 5.1 Policy 封装

Policy 是推理的核心封装：

```python
class Policy(BasePolicy):
    def __init__(self, model, transforms, output_transforms, sample_kwargs):
        ...
    
    def infer(self, obs: dict) -> dict:
        inputs = self._input_transform(obs)     # 应用输入变换
        # JAX: 转 jax.Array + 加 batch 维度
        # PyTorch: 转 torch.Tensor + 移到 GPU
        observation = Observation.from_dict(inputs)
        actions = model.sample_actions(rng, observation, **sample_kwargs)
        outputs = self._output_transform(outputs) # 输出反变换
        return {"actions": ..., "state": ...}
```

### 5.2 创建训练好的策略

```python
from openpi.policies import policy_config
from openpi.training import config as _config

config = _config.get_config("pi05_droid")
policy = policy_config.create_trained_policy(
    config,
    "gs://openpi-assets/checkpoints/pi05_droid",
    default_prompt="pick up the blue block",
)
```

### 5.3 WebSocket 服务

```
┌─────────────┐          WebSocket           ┌─────────────┐
│  机器人端    │ ◄══════════════════════════► │  GPU 服务器  │
│  (轻量)     │   obs → action_chunk         │  推理服务    │
│  openpi-   │                               │  8000 端口  │
│  client    │                               │  serve_     │
│            │                               │  policy.py  │
└─────────────┘                               └─────────────┘
```

启动服务：
```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=checkpoints/pi05_droid/my_exp/20000
```

客户端使用：
```python
from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy("ws://server:8000")
result = policy.infer({
    "image": {"base_0_rgb": camera_image},
    "state": robot_joint_positions,
})
actions = result["actions"]  # (action_horizon, action_dim)
```

### 5.4 推理采样参数

`model.sample_actions` 的关键参数：

```python
# pi0/pi0.5 流匹配采样
model.sample_actions(
    rng,
    observation,
    num_steps=10,     # 去噪步数（越大越精确，但越慢）
    noise=None,       # 可以传入自定义噪声种子
)

# pi0-FAST 自回归采样
model.sample_actions(
    rng,
    observation,
    max_decoding_steps=256,  # 最大生成长度
    temperature=0.0,         # 采样温度（0=贪心）
)
```

---

## 第六部分：迁移学习（Fine-tuning）

### 6.1 迁移学习流程

在你的机器人数据集上微调 pi0.5 的完整流程：

```
1. 准备数据
   └─ 收集你的机器人数据，转换为 LeRobot 格式
   └─ 确保包含: images, state, actions, prompt(可选)
   
2. 创建数据转换脚本
   └─ 定义你的机器人平台的输入/输出转换
   └─ 参考 examples/aloha_real/ 或 examples/droid/

3. 注册新的 TrainConfig
   └─ 在 config.py 中添加你的配置
   └─ 选择模型（pi0/pi0.5/pi0-FAST）
   └─ 配置数据源和变换

4. 计算归一化统计量
   └─ uv run scripts/compute_norm_stats.py --config-name <你的配置>

5. 训练
   └─ JAX: uv run scripts/train.py <配置>
   └─ 或 PyTorch: torchrun ... train_pytorch.py

6. 评估
   └─ 启动推理服务测试
```

### 6.2 以 LIBERO 为例的微调配置

```python
TrainConfig(
    name="pi05_libero",
    model=pi0_config.Pi0Config(
        pi05=True,               # pi0.5 模式
        action_horizon=10,       # 动作预测长度
        discrete_state_input=False, # LIBERO 不用离散状态
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",  # LeRobot 数据集
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000, peak_lr=5e-5,
        decay_steps=1_000_000, decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=30_000,
)
```

### 6.3 LoRA 微调（低内存）

LoRA 通过在注意力层和 FFN 层插入低秩矩阵来大幅减少可训练参数量：

```python
Pi0Config(
    paligemma_variant="gemma_2b_lora",       # 主模型用 LoRA
    action_expert_variant="gemma_300m_lora",  # Action expert 也用 LoRA
)
```

LoRA 配置在 gemma.py 中定义：
```python
lora_configs={
    "attn": LoRAConfig(rank=16, alpha=16.0),  # 注意力 LoRA
    "ffn": LoRAConfig(rank=16, alpha=16.0),   # FFN LoRA
}
```

冻结非 LoRA 参数：
```python
freeze_filter = Pi0Config(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
).get_freeze_filter()
```

### 6.4 添加新机器人的步骤（通用模板）

以添加一个新的单臂机器人为例，你需要：

**A. 创建机器人策略转换**（仿照 `droid_policy.py`）：

```python
# my_robot_policy.py

@dataclasses.dataclass(frozen=True)
class MyRobotInputs(DataTransformFn):
    """将原始传感器数据映射到模型输入格式"""
    model_type: ModelType
    
    def __call__(self, data: DataDict) -> DataDict:
        # 重映射图像键名
        image = {
            "base_0_rgb": data["observation/image"],
            "left_wrist_0_rgb": data["observation/wrist_image"],
        }
        state = data["observation/joint_positions"]
        return {"image": image, "state": state, ...}
```

**B. 注册新配置**：

```python
TrainConfig(
    name="pi05_my_robot",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=16),
    data=SimpleDataConfig(
        repo_id="your-org/my_dataset",
        assets=AssetsConfig(asset_id="my_robot"),
        data_transforms=lambda model: Group(
            inputs=[MyRobotInputs(model_type=model.model_type)],
            outputs=[MyRobotOutputs()],
        ),
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=20_000,
    batch_size=32,
)
```

**C. 可选的 action 转换**（是否需要 delta actions）：

```python
# 如果数据集使用绝对关节角度，需要转为 delta
delta_action_mask = make_bool_mask(6, -1)  # 前6维转delta，最后一维(gripper)不动
data_transforms = data_transforms.push(
    inputs=[DeltaActions(delta_action_mask)],
    outputs=[AbsoluteActions(delta_action_mask)],
)
```

---

## 第七部分：关键代码定位

### 7.1 如果你想修改模型架构

| 修改目标 | 文件 |
|---------|------|
| 修改模型配置 | [pi0_config.py](../src/openpi/models/pi0_config.py) |
| 修改流匹配损失/采样 | [pi0.py](../src/openpi/models/pi0.py) 的 `compute_loss` / `sample_actions` |
| 修改 Gemma 语言模型 | [gemma.py](../src/openpi/models/gemma.py) |
| 修改视觉编码器 | [siglip.py](../src/openpi/models/siglip.py) |
| 添加新的模型变体 | 继承 `BaseModelConfig` 并实现 `create()` |
| PyTorch 模型修改 | [pi0_pytorch.py](../src/openpi/models_pytorch/pi0_pytorch.py) |

### 7.2 如果你想修改训练流程

| 修改目标 | 文件 |
|---------|------|
| JAX 训练循环 | [train.py](../scripts/train.py) |
| PyTorch 训练循环 | [train_pytorch.py](../scripts/train_pytorch.py) |
| 数据加载 | [data_loader.py](../src/openpi/training/data_loader.py) |
| 优化器 / LR | [optimizer.py](../src/openpi/training/optimizer.py) |
| 检查点管理 | [checkpoints.py](../src/openpi/training/checkpoints.py) |
| FSDP 配置 | [sharding.py](../src/openpi/training/sharding.py) |

### 7.3 如果你想修改数据处理

| 修改目标 | 文件 |
|---------|------|
| 数据变换（通用） | [transforms.py](../src/openpi/transforms.py) |
| 机器人输入/输出 | [droid_policy.py](../src/openpi/policies/droid_policy.py) 等 |
| 分词器 | [tokenizer.py](../src/openpi/models/tokenizer.py) |
| 归一化 | [normalize.py](../src/openpi/shared/normalize.py) |
| 图像处理 | [image_tools.py](../src/openpi/shared/image_tools.py) |

---

## 第八部分：常见问题排查

### Q: 训练时 OOM（显存不足）
- 减小 `batch_size`
- 启用 FSDP：`--fsdp_devices 2`（2 卡分片）
- 使用 LoRA 微调
- JAX 设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`

### Q: 找不到归一化统计量
- 运行 `uv run scripts/compute_norm_stats.py --config-name <你的配置>`
- 检查 `assets_base_dir` 路径是否正确

### Q: 数据集格式不匹配
- 用 `RepackTransform` 重映射键名
- 确保图像在 `data["image"]` 字典中
- 确保 state 和 actions 存在

### Q: prompt 未生效
- 如使用 LeRobot，设置 `prompt_from_task=True`
- 或在推理时传 `default_prompt`
- pi0.5 的 prompt 格式为 `"Task: {task}, State: {state};\nAction: "`

### Q: Pi0.5 训练时 loss nan
- 检查 `clip_gradient_norm` 设置
- 确认输入数据范围正确（图像在 [-1,1]，state/actions 已归一化）
- 减小学习率

### Q: 推理结果不理想
- 增加 `num_steps`（去噪步数）
- 检查归一化统计量是否与训练时一致
- 确认 action_horizon 设置是否正确

---

## 第九部分：学习路线图

建议按以下顺序学习：

```
第1步：环境搭建
  └─ 安装 uv、clone 代码、uv sync
  └─ 运行 debug 配置验证环境

第2步：跑通推理
  └─ 下载预训练 checkpoint
  └─ 用 pi05_droid 配置运行推理
  └─ 理解 Policy 封装流程

第3步：理解配置系统
  └─ 阅读 config.py 的所有配置
  └─ 理解每个字段的含义
  └─ 尝试修改参数看效果

第4步：数据管线
  └─ 理解三阶段变换
  └─ 理解 pi0 vs pi0.5 的 tokenizer 差异
  └─ 理解归一化的作用

第5步：模型架构
  └─ 阅读 pi0.py：embed_prefix, embed_suffix, compute_loss, sample_actions
  └─ 理解 pi0 vs pi0.5 差异
  └─ 对比 PyTorch 版本的实现

第6步：训练
  └─ compute_norm_stats
  └─ JAX 训练流程
  └─ 监控训练指标

第7步：迁移学习
  └─ 将自己的数据转为 LeRobot 格式
  └─ 创建新配置
  └─ 全量微调和 LoRA 微调

第8步：部署
  └─ WebSocket 服务
  └─ 客户端集成
  └─ Docker 部署
```
