# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`openpi` is Physical Intelligence's open-source robotics repo implementing vision-language-action (VLA) models: π₀ (flow-based), π₀-FAST (autoregressive), and π₀.₅ (upgraded with knowledge insulation). Provides base model checkpoints, fine-tuning pipelines, and inference servers for ALOHA, DROID, LIBERO, and UR5 robot platforms.

## Build & Dependencies

```bash
# Setup (requires uv)
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# PyTorch path requires manual transformers patches:
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

# Docker alternative: see docs/docker.md
```

## Test, Lint, Format

```bash
# Tests (CI uses --strict-markers -m "not manual")
uv run pytest --strict-markers -m "not manual"
uv run pytest src/openpi/models/model_test.py   # specific test

# Lint & format (ruff, line-length 120)
ruff check .
ruff format .
ruff check . --fix
```

## Training

Configs are registered in [src/openpi/training/config.py](src/openpi/training/config.py). Use `get_config("name")` to retrieve them.

```bash
# 1. Compute norm stats first
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 2. JAX train
uv run scripts/train.py <config_name> --exp-name=my_run --overwrite

# 3. PyTorch train (single or multi-GPU via torchrun)
uv run scripts/train_pytorch.py <config_name> --exp_name my_run
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py <config_name> --exp_name my_run
```

## Inference/Serving

```bash
# Serve a trained policy
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=<checkpoint_dir>

# Client (works over WebSocket)
from openpi_client import websocket_client_policy
policy = websocket_client_policy.WebsocketClientPolicy("ws://server:8000")
action_chunk = policy.infer(observation)["actions"]
```

## Architecture

### Dual Backend
- **JAX** (`src/openpi/models/`): Primary training framework (Flax NNX, used for full training)
- **PyTorch** (`src/openpi/models_pytorch/`): Newer, supports π₀/π₀.₅ inference + DDP training. Gaps: no FAST, no FSDP, no LoRA, no EMA.

### Model Types
- **π₀**: Flow-based VLA (Paligemma 2B vision-language + 300M action expert). Uses continuous state input, flow-matching action head.
- **π₀-FAST**: Autoregressive VLA with FAST action tokenizer. Discrete tokenized actions with causal attention.
- **π₀.₅**: π₀ upgrade with discrete state tokens, adaRMSNorm in action expert, knowledge insulation. Configs use `pi05=True`.

### Key Source Layout

| Path | Role |
|------|------|
| `src/openpi/models/` | JAX model definitions (Pi0, Pi0FAST) with NNX |
| `src/openpi/models/pi0_config.py` | Pi0Config dataclass — controls model architecture choices |
| `src/openpi/training/config.py` | Central `_CONFIGS` registry — all TrainConfigs defined here |
| `src/openpi/training/data_loader.py` | Unified data loader (LeRobot + RLDS) |
| `src/openpi/transforms.py` | Data transform pipeline (repack → robot → model) |
| `src/openpi/policies/` | Policy wrappers + robot-specific I/O transforms |
| `src/openpi/policies/policy_config.py` | `create_trained_policy()` — auto-detects JAX vs PyTorch |
| `src/openpi/serving/websocket_policy_server.py` | WebSocket inference server (port 8000) |
| `packages/openpi-client/` | Lightweight inference-only client (uv workspace member) |

### Data Pipeline

Three-stage transform flow in `transforms.py`:
1. **repack_transforms**: dataset format → common dict format
2. **data_transforms**: robot-specific (ALOHA joint→delta, DROID joint→velocity, etc.)
3. **model_transforms**: model-specific (resize images 224×224, tokenize, pad)

Output transforms reverse the pipeline (unnormalize, convert actions).

### Key Conventions

- **Config-driven**: Add new robot/dataset by registering a `TrainConfig` in [config.py](src/openpi/training/config.py)
- **Import aliases**: `import openpi.models.model as _model`, `import openpi.transforms as _transforms`, `import openpi.shared.array_typing as at`
- **jaxtyping**: Array shapes annotated with `at.Float[ArrayT, "*b h w c"]` style — runtime-checked via `@at.typecheck`
- **Norm stats prerequisite**: Always run `compute_norm_stats.py` before training
- **Both backends**: Model/training changes often need updates in both `models/` (JAX) and `models_pytorch/` (PyTorch)
- **No exceptions/RTTI** in model code — standard Python
- **Manual test marker**: Large/download tests marked `@pytest.mark.manual`
- **GS checkpoints**: Base checkpoints at `gs://openpi-assets/checkpoints/`, cached to `~/.cache/openpi` (`OPENPI_DATA_HOME` overrides)
