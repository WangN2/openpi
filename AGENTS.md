# openpi — Agent Guide

This document provides the essential context an AI coding agent needs to work effectively in the `openpi` repository. It is derived from the actual project structure, configuration, and conventions — not from assumptions.

---

## Project Overview

`openpi` is the open-source robotics repository from **Physical Intelligence**. It implements vision-language-action (VLA) models for robot control, including:

- **π₀ (pi0)**: Flow-based VLA model.
- **π₀-FAST**: Autoregressive VLA using the FAST action tokenizer.
- **π₀.₅ (pi05)**: Upgraded π₀ with improved open-world generalization via knowledge insulation.

The repo provides base model checkpoints (pre-trained on 10k+ hours of robot data), fine-tuning pipelines, inference servers, and example integrations for ALOHA, DROID, LIBERO, and UR5 robot platforms.

**Repository**: https://github.com/Physical-Intelligence/openpi  
**License**: See `LICENSE` and `LICENSE_GEMMA.txt`.

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Package Manager** | `uv` (Astral) — lock file is `uv.lock` |
| **Build Backend** | `hatchling` |
| **Python** | >= 3.11 (main project); `packages/openpi-client` supports >= 3.7 |
| **ML Frameworks** | JAX + Flax (NNX) (`jax[cuda12]==0.5.3`, `flax==0.10.2`) AND PyTorch (`torch==2.7.1`) |
| **Checkpointing** | `orbax-checkpoint` (JAX), `safetensors` (PyTorch) |
| **VLM/LLM** | `transformers==4.53.2`, `sentencepiece` |
| **Data** | LeRobot (git dependency), RLDS via `dlimp` (git dependency), `tensorflow-cpu==2.15.0` |
| **Logging** | `wandb`, `tqdm-loggable` |
| **Typing** | `jaxtyping`, `beartype`, `numpydantic` |
| **Lint/Format** | `ruff` |
| **Test** | `pytest` |
| **Pre-commit** | `pre-commit` with `uv-lock` and `ruff` hooks |
| **Docker** | NVIDIA CUDA 12.2 runtime images, Docker Compose for examples |

---

## Project Structure

```
openpi/
├── pyproject.toml              # Main package config, uv workspace, ruff, pytest
├── uv.lock                     # Reproducible dependency lock
├── .python-version             # Python version pin
├── src/openpi/                 # Main source package
│   ├── models/                 # JAX model implementations
│   │   ├── model.py            # Base model abstractions (ModelType, Observation, Actions)
│   │   ├── pi0.py              # π₀ and π₀.₅ flow-matching model
│   │   ├── pi0_fast.py         # π₀-FAST autoregressive model
│   │   ├── pi0_config.py       # Model configuration dataclasses
│   │   ├── gemma.py / gemma_fast.py / siglip.py / vit.py
│   │   ├── tokenizer.py        # PaligemmaTokenizer, FASTTokenizer
│   │   ├── lora.py             # Low-rank adaptation for fine-tuning
│   │   └── *_test.py           # Unit tests
│   ├── models_pytorch/         # PyTorch model implementations
│   │   ├── pi0_pytorch.py      # PyTorch π₀/π₀.₅ model
│   │   ├── gemma_pytorch.py    # PyTorch PaliGemma + Expert wrapper
│   │   ├── preprocessing_pytorch.py
│   │   └── transformers_replace/  # Patches for transformers==4.53.2
│   ├── policies/               # Policy wrappers + robot-specific transforms
│   │   ├── policy.py           # Policy class (JAX + PyTorch backends)
│   │   ├── policy_config.py    # create_trained_policy() — auto-detects JAX vs PyTorch
│   │   ├── aloha_policy.py / droid_policy.py / libero_policy.py
│   │   └── policy_test.py
│   ├── training/               # Training infrastructure
│   │   ├── config.py           # Central config registry (_CONFIGS)
│   │   ├── train.py            # JAX training entrypoint
│   │   ├── train_pytorch.py    # PyTorch DDP training entrypoint
│   │   ├── data_loader.py      # Unified data loader (LeRobot + RLDS)
│   │   ├── checkpoints.py      # Checkpoint save/load with orbax
│   │   ├── optimizer.py        # AdamW + cosine LR schedules
│   │   ├── weight_loaders.py   # Base model weight loading utilities
│   │   ├── sharding.py         # JAX FSDP configuration
│   │   ├── utils.py            # Training utilities
│   │   ├── droid_rlds_dataset.py
│   │   └── *_test.py
│   ├── serving/
│   │   └── websocket_policy_server.py   # WebSocket policy server for remote inference
│   ├── shared/                 # Utilities
│   │   ├── array_typing.py     # JAX typing annotations with jaxtyping
│   │   ├── download.py         # GCS download with local caching
│   │   ├── normalize.py        # Normalization stats
│   │   ├── image_tools.py      # Image resize/pad utilities
│   │   ├── nnx_utils.py        # Flax NNX helpers
│   │   └── *_test.py
│   ├── transforms.py           # Data transform pipeline (Normalize, Tokenize, ResizeImages, etc.)
│   ├── conftest.py             # Pytest configuration/fixtures
│   └── py.typed                # PEP 561 marker
├── packages/openpi-client/     # Lightweight client package (uv workspace member)
│   ├── pyproject.toml          # Minimal deps: numpy, pillow, websockets, msgpack, dm-tree
│   ├── base_policy.py
│   ├── websocket_client_policy.py
│   ├── msgpack_numpy.py
│   ├── image_tools.py
│   └── runtime/                # Agent/runtime framework
├── scripts/
│   ├── train.py                # JAX training launcher
│   ├── train_pytorch.py        # PyTorch training launcher
│   ├── serve_policy.py         # Policy server launcher
│   ├── compute_norm_stats.py   # Compute normalization statistics
│   ├── train_test.py           # End-to-end training tests
│   └── docker/                 # Dockerfiles and compose files
├── examples/
│   ├── aloha_real/ / aloha_sim/ / droid/ / libero/ / ur5/
│   ├── simple_client/          # Minimal inference test client
│   ├── convert_jax_model_to_pytorch.py
│   ├── inference.ipynb
│   └── policy_records.ipynb
├── docs/
│   ├── docker.md               # Docker setup instructions
│   ├── remote_inference.md     # Remote inference guide
│   └── norm_stats.md           # Normalization statistics guide
├── third_party/                # Git submodules
│   ├── aloha/                  # Physical-Intelligence/aloha
│   └── libero/                 # Lifelong-Robot-Learning/LIBERO
└── .github/workflows/
    ├── test.yml                # PR test runner (self-hosted runner)
    └── pre-commit.yml          # Lint/format checks
```

---

## Build and Development Setup

### Prerequisites

- Ubuntu 22.04 (tested; other OSes not currently supported).
- NVIDIA GPU with adequate VRAM (see README for memory requirements per mode).
- `git` with `git-lfs` support.
- `uv` installed: https://docs.astral.sh/uv/getting-started/installation/

### Initial Setup

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# If already cloned:
git submodule update --init --recursive

# Sync dependencies (GIT_LFS_SKIP_SMUDGE=1 is required because LeRobot uses git-lfs)
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Optional: install editable
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### PyTorch-Specific Setup

The PyTorch path requires manually patching the installed `transformers` library:

```bash
# Verify version
uv pip show transformers   # must be 4.53.2

# Apply patches
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

**Warning**: With uv's default hardlink mode, this permanently affects the transformers package in the uv cache. To undo, run `uv cache clean transformers`.

---

## Build and Test Commands

### Running Tests

```bash
# Run all tests (excludes manual tests)
uv run pytest --strict-markers -m "not manual"

# Run tests in a specific directory
uv run pytest src/openpi/models/

# Run with GPU detection fallback (conftest.py auto-sets JAX_PLATFORMS=cpu if no GPU)
uv run pytest src/openpi/models/model_test.py
```

### Linting and Formatting

```bash
# Check
ruff check .

# Format
ruff format .

# Fix auto-fixable issues
ruff check . --fix
```

### Pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

---

## Code Style Guidelines

### Formatter and Linter

- **Tool**: `ruff` (line length **120**, target Python **3.11**).
- **Excluded paths**: `docker/`, `third_party/`, `src/openpi/models_pytorch/transformers_replace/*`.

### Import Style

- **Single-line imports** (`force-single-line = true` in ruff isort config).
- **Standard library** first, **third-party** second, **internal** third.
- Internal imports commonly use aliases to avoid name collisions:
  ```python
  import openpi.models.model as _model
  import openpi.shared.array_typing as at
  import openpi.transforms as _transforms
  ```
- Use `collections.abc` imports (`Sequence`, `Mapping`, `Callable`) over `typing` counterparts.

### Typing

- Heavy use of **Python 3.11+ union syntax**: `str | None`.
- **jaxtyping** for array shape/dtype annotations:
  ```python
  at.Float[ArrayT, "*b h w c"]
  at.Int[at.Array, "*b l"]
  ```
- Runtime type checking via a custom `typecheck` decorator (wraps `jaxtyped` + `beartype`).
- Use `typing_extensions.override` when overriding methods.
- Type variables and generics are used where appropriate.

### Naming and Patterns

- **Classes**: `PascalCase`.
- **Functions/variables**: `snake_case`.
- **Constants**: `UPPER_SNAKE_CASE`.
- **Private/internal aliases**: prefixed with `_`.
- **Dataclasses**: heavily used for configs and transforms; often `frozen=True`.
- **Logging**: use `logging.getLogger("openpi")`, not the root logger.

### Docstrings

- Google/NumPy-like style with `Args:` and `Returns:` sections.
- Public functions and classes should have descriptive docstrings.

---

## Testing Instructions

### Test Configuration

- **Framework**: `pytest`
- **Config location**: `pyproject.toml` (`[tool.pytest.ini_options]`)
- **Test paths**: `src`, `scripts`, `packages`
- **Custom marker**: `manual` — tests that should NOT run in CI (e.g., downloading large checkpoints).

### Test File Locations

Tests live alongside source code in `*_test.py` files:

| File | Coverage |
|------|----------|
| `src/openpi/models/model_test.py` | Model creation, loss shapes, action shapes, checkpoint restore |
| `src/openpi/models/pi0_test.py` | π₀-specific logic |
| `src/openpi/models/tokenizer_test.py` | Tokenizers |
| `src/openpi/models/lora_test.py` | LoRA implementation |
| `src/openpi/policies/policy_test.py` | Policy inference logic |
| `src/openpi/training/data_loader_test.py` | Data loading |
| `src/openpi/transforms_test.py` | Data transforms |
| `src/openpi/shared/*_test.py` | Download, image_tools, normalize |
| `scripts/train_test.py` | End-to-end training + resuming |
| `packages/openpi-client/src/openpi_client/*_test.py` | Client utilities |

### GPU-Aware Testing

`src/openpi/conftest.py` detects GPU availability via `pynvml`. If no GPU is found, it automatically sets `JAX_PLATFORMS=cpu` so tests can run on CPU.

### CI

- **Test workflow** (`.github/workflows/test.yml`): Runs on `openpi-verylarge` self-hosted runner on every PR. Installs FFmpeg, uv, syncs deps, runs `uv run pytest --strict-markers -m "not manual"`.
- **Pre-commit workflow** (`.github/workflows/pre-commit.yml`): Runs on push to `main` and PRs.

---

## Runtime Architecture

### Training

#### JAX Training

```bash
# Compute norm stats first
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# Train (single GPU)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite

# Train with FSDP sharding across multiple GPUs
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --fsdp-devices 2
```

- Supports FSDP (Fully Sharded Data Parallelism) via `jax.sharding.Mesh`.
- Checkpointing with `orbax` (save/resume supported).
- EMA, Weights & Biases logging, mixed precision (bfloat16 activations, float32 weights).

#### PyTorch Training

```bash
# Single GPU
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Multi-GPU single node
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<N> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Multi-node
uv run torchrun --nnodes=<N> --nproc_per_node=<G> --node_rank=<R> --master_addr=<IP> --master_port=<P> \
    scripts/train_pytorch.py <config_name> --exp_name <run_name>
```

- Supports DDP via `torchrun`.
- Uses `safetensors` for model checkpoints.
- **Not yet supported**: FSDP, LoRA, EMA, mixed precision, π₀-FAST.

### Inference and Serving

#### Policy Server

```bash
# Serve a pre-trained checkpoint
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000

# Serve default environment policies
uv run scripts/serve_policy.py --env droid
```

- Listens on port **8000** by default.
- Serves policies over **WebSockets**.
- Health check at `/healthz`.
- Supports `--record` for debugging policy behavior.

#### Remote Inference

The heavy model runs on a GPU server; the robot uses a minimal WebSocket client:

```python
from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy("ws://server:8000")
action_chunk = policy.infer(observation)["actions"]
```

The client resizes images to 224x224 before sending to minimize bandwidth.

### Data Pipeline

Data flows through three transform stages:

1. **repack_transforms** — dataset-specific format → common format.
2. **data_transforms** — robot-specific transforms (ALOHA, DROID, LIBERO).
3. **model_transforms** — model-specific (resize images to 224x224, tokenize prompts, pad states/actions).

Output transforms reverse this pipeline.

---

## Deployment

### Docker

- **Base policy server image**: `scripts/docker/serve_policy.Dockerfile`
  - Based on `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04`.
  - Installs uv, Python 3.11.9, syncs deps from `uv.lock`, applies `transformers_replace` patches.
  - Default command: `uv run scripts/serve_policy.py $SERVER_ARGS`.
- **Docker Compose**: `scripts/docker/compose.yml` runs the server with host networking and GPU reservation.
- **Example images**: Each example (`aloha_real`, `droid`, `libero`, `simple_client`) has its own `Dockerfile` and `compose.yml`.

### Important Docker Notes

- Docker must be in **rootless mode** with **NVIDIA Container Toolkit** for GPU access.
- Docker Desktop and snap-installed Docker are **explicitly incompatible**.
- Setup scripts provided: `scripts/docker/install_docker_ubuntu22.sh` and `install_nvidia_container_toolkit.sh`.

---

## Security Considerations

- **Transformers patches**: The PyTorch path requires copying patched files into the installed `transformers` package. This modifies a shared dependency and, with uv's default hardlink mode, can affect the global uv cache. Be aware that this is a non-standard and potentially cache-polluting operation.
- **GCS assets**: Checkpoints and norm stats are downloaded from `gs://openpi-assets` and cached locally in `~/.cache/openpi` (or `OPENPI_DATA_HOME`). Ensure this path is secure if running in multi-user environments.
- **WebSocket serving**: The policy server uses WebSockets without built-in authentication. In production deployments, place it behind a secure network boundary or add authentication at the infrastructure level.
- **Git submodules**: The repo depends on `third_party/aloha` and `third_party/libero` submodules. Ensure these are from trusted sources and updated securely.

---

## Key Conventions for Agents

1. **Dual backend awareness**: Any model or training change may need corresponding updates in both `src/openpi/models/` (JAX) and `src/openpi/models_pytorch/` (PyTorch). PyTorch support is newer and has some gaps (no FAST, no FSDP, no LoRA, no EMA).

2. **Config-driven**: Most behavior is controlled through `src/openpi/training/config.py`. When adding a new robot or dataset, define a `TrainConfig` and `DataConfig` there.

3. **Transforms are central**: Data processing logic lives in `src/openpi/transforms.py` and robot-specific policy modules. Changes to input/output formats usually require updating both training transforms and policy transforms.

4. **Norm stats prerequisite**: Training requires precomputed normalization statistics. Run `scripts/compute_norm_stats.py` before starting training.

5. **Tests should avoid `manual` marker**: If a test downloads large files or requires specific hardware, mark it `@pytest.mark.manual`. Otherwise, keep it runnable in CI.

6. **Ruff formatting is mandatory**: The project enforces ruff format and lint via pre-commit and CI. Always run `ruff check . --fix && ruff format .` before committing.

7. **uv lock file must stay in sync**: The pre-commit hook `uv-lock` verifies `uv.lock` is up to date with `pyproject.toml`. After changing dependencies, run `uv lock`.
