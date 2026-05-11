"""
第2步：推理流程演示
===================

目标：理解从 配置 → 模型加载 → Policy封装 → 推理 的完整流程。

先用 debug 模型（小模型、无需下载）跑通流程，再切换到真实 checkpoint。
"""

import logging
import numpy as np
from pprint import pprint

logging.basicConfig(level=logging.INFO)

# ═══════════════════════════════════════════════════════════
# 1. 配置系统：一切从这里开始
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("1. 加载配置")
print("=" * 60)

from openpi.training import config as _config

# debug 配置使用 dummy 模型（非常小，适合验证流程）
config = _config.get_config("debug")
print(f"   配置名称: {config.name}")
print(f"   模型类型: {config.model.model_type}")
print(f"   动作维度: {config.model.action_dim}")
print(f"   动作预测长度 (action_horizon): {config.model.action_horizon}")
print(f"   最大 token 长度: {config.model.max_token_len}")
print(f"   模型变体: {config.model.paligemma_variant}, {config.model.action_expert_variant}")
print()

# ═══════════════════════════════════════════════════════════
# 2. 数据配置：了解数据来源和变换管线
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("2. 数据配置（了解数据变换流程）")
print("=" * 60)

data_config = config.data.create(config.assets_dirs, config.model)
print(f"   数据集: {data_config.repo_id}")
print(f"   使用归一化: {data_config.norm_stats is not None}")
print(f"   归一化方式: {'quantile' if data_config.use_quantile_norm else 'z-score'}")

# 查看 transform 管线
print("\n   Transform 管线（数据→模型输入的完整链条）:")
for i, t in enumerate(data_config.repack_transforms.inputs):
    print(f"     repack [{i}]: {type(t).__name__}")
for i, t in enumerate(data_config.data_transforms.inputs):
    print(f"     data   [{i}]: {type(t).__name__}")
for i, t in enumerate(data_config.model_transforms.inputs):
    print(f"     model  [{i}]: {type(t).__name__}")
print()

# ═══════════════════════════════════════════════════════════
# 3. 创建模型：理解模型结构
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("3. 查看模型输入输出规格")
print("=" * 60)

obs_spec, act_spec = config.model.inputs_spec(batch_size=1)
print(f"\n   观察空间 (Observation):")
print(f"     图像: {', '.join(obs_spec.images.keys())}")
for k, v in obs_spec.images.items():
    print(f"       {k}: shape={v.shape}, dtype={v.dtype.name}")
print(f"     状态 (state): shape={obs_spec.state.shape}, dtype={obs_spec.state.dtype.name}")
if obs_spec.tokenized_prompt is not None:
    print(f"     prompt tokens: shape={obs_spec.tokenized_prompt.shape}, dtype={obs_spec.tokenized_prompt.dtype.name}")
print(f"\n   动作空间 (Actions):")
print(f"     shape={act_spec.shape}, dtype={act_spec.dtype.name}")
print(f"     含义: (batch=1, action_horizon={config.model.action_horizon}, action_dim={config.model.action_dim})")
print()

# ═══════════════════════════════════════════════════════════
# 4. 创建策略（Policy）：完整的推理封装
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("4. 创建 Policy（推理的核心封装）")
print("=" * 60)

from openpi.policies import policy_config

# create_trained_policy 的流程：
#   1. 创建模型（随机初始化或加载 checkpoint）
#   2. 加载数据配置（transforms, norm stats）
#   3. 把 transforms 和 model 组合成 Policy
#   4. Policy.infer(obs) = transforms(obs) → model.sample_actions → output_transforms
policy = policy_config.create_trained_policy(
    config,
    "./checkpoints/debug/debug/9",  # 用我们刚才训练的 debug 检查点
    default_prompt="do something",  # 没有提供 prompt 时的默认值
)

print(f"   策略元数据: {policy.metadata}")
print(f"   采样参数: {policy._sample_kwargs}")
print()

# ═══════════════════════════════════════════════════════════
# 5. 执行推理！理解数据流
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("5. 构造输入并执行推理")
print("=" * 60)

# 模型期望的输入格式（在 policy.infer 中会自动经过 transforms 处理）：
#   obs = {
#       "image": {
#           "base_0_rgb": np.ndarray [H, W, 3],
#           "left_wrist_0_rgb": np.ndarray [H, W, 3],
#           "right_wrist_0_rgb": np.ndarray [H, W, 3],
#       },
#       "state": np.ndarray [action_dim],
#       "prompt": str (可选),
#   }

# 构造随机输入（debug 模型不关心实际内容）
fake_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
fake_state = np.random.randn(config.model.action_dim).astype(np.float32)

obs = {
    "image": {
        "base_0_rgb": fake_image,
        "left_wrist_0_rgb": fake_image,
        "right_wrist_0_rgb": fake_image,
    },
    "state": fake_state,
    # 注意：我们没有传 "prompt"，policy 会使用 default_prompt
}

print("   输入:")
print(f"     图像: 3 个视角, 每个 224x224x3")
print(f"     状态维度: {fake_state.shape}")
print()

result = policy.infer(obs)

print("   输出:")
print(f"     动作: shape={result['actions'].shape}")
print(f"       actions[0,:5] (前5维): {result['actions'][0, :5]}")
print(f"     推理耗时: {result['policy_timing']['infer_ms']:.1f} ms")
print()

# ═══════════════════════════════════════════════════════════
# 6. 深入理解 Policy.infer 内部流程
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("6. Policy.infer 内部发生了什么？")
print("=" * 60)

# 让我们手动复现 Policy.infer 的流程，理解每一步

inputs = obs.copy()  # 复制输入

# Step A: 输入变换（input_transform）
print("\n   Step A: 应用输入变换")
print(f"     原始 keys: {list(inputs.keys())}")
transformed = policy._input_transform(inputs)
print(f"     变换后 keys: {list(transformed.keys())}")

# Step B: 模型推理（sample_actions）
# 注意：这里在 Policy 内部还会发生：
#   1. numpy → jax.Array 转换
#   2. 添加 batch 维度
#   3. 调用 model.sample_actions
#   4. 去掉 batch 维度
#   5. jax.Array → numpy 转换
print(f"\n   Step B: 模型推理")

# Step C: 输出变换（output_transform）
print(f"\n   Step C: 输出变换")
print(f"     输出 keys: {list(result.keys())}")

# ═══════════════════════════════════════════════════════════
# 7. 切换到真实配置会怎样？
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("7. 切换到真实配置（pi05_droid）")
print("=" * 60)

real_config = _config.get_config("pi05_droid")
print(f"\n   配置: {real_config.name}")
print(f"   模型: pi0.5 (pi05={real_config.model.pi05})")
print(f"   action_dim={real_config.model.action_dim}")
print(f"   action_horizon={real_config.model.action_horizon}")
print(f"   max_token_len={real_config.model.max_token_len}")
print(f"   模型变体: {real_config.model.paligemma_variant}")
print(f"   action expert: {real_config.model.action_expert_variant}")

real_obs_spec, real_act_spec = real_config.model.inputs_spec(batch_size=1)
print(f"\n   真实模型输入规格:")
for k, v in real_obs_spec.images.items():
    print(f"     图像 {k}: shape={v.shape}")
print(f"     状态: shape={real_obs_spec.state.shape}")
print(f"     动作输出: shape={real_act_spec.shape}")

print(f"\n   使用真实推理:")
print(f"     policy = policy_config.create_trained_policy(")
print(f"         _config.get_config('pi05_droid'),")
print(f"         'gs://openpi-assets/checkpoints/pi05_droid',")
print(f"     )")
print(f"     result = policy.infer(obs)")
print()

print("✅ 推理流程演示完成！")
print("关键要点：")
print("  - 一切从 TrainConfig 开始，配置驱动")
print("  - Policy 封装了 transforms + model + output_transforms")
print("  - 输入是 dict(images, state, prompt)，输出是 dict(actions)")
print("  - 输入数据会经过三阶段变换后送入模型")
print("  - debug→真实模型切换只需要换 config 和 checkpoint 路径")
