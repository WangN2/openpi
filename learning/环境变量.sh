#!/usr/bin/env bash

# 在 /ephstorage/openpi 仓库根目录执行：source learning/环境变量.sh
export OPENPI_DATA_HOME=/ephstorage/openpi_data/cache/openpi
export HF_LEROBOT_HOME=/ephstorage/openpi_data/lerobot
export HF_DATASETS_CACHE=/home/che.lin/.cache/huggingface/datasets
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/ephstorage/openpi/third_party/libero:${PYTHONPATH:-}
export LIBERO_CONFIG_PATH=/ephstorage/openpi_data/runtime/libero-config

_OPENPI_MESA_LIB=/ephstorage/openpi_data/runtime/mesa-egl/usr/lib/x86_64-linux-gnu
_OPENPI_EGL_LIB=/ephstorage/openpi_data/runtime/libegl1/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH="${_OPENPI_EGL_LIB}:${_OPENPI_MESA_LIB}:${LD_LIBRARY_PATH:-}"
export LIBGL_DRIVERS_PATH="${_OPENPI_MESA_LIB}/dri"
export __EGL_VENDOR_LIBRARY_FILENAMES=/ephstorage/openpi_data/runtime/mesa-egl/usr/share/glvnd/egl_vendor.d/50_mesa.json
export LIBGL_ALWAYS_SOFTWARE=1
export EGL_PLATFORM=surfaceless
unset _OPENPI_MESA_LIB _OPENPI_EGL_LIB
