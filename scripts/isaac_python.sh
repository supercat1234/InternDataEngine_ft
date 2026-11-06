#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${INTERNDATA_HOME:-$REPO_DIR/../.interndata}"
export ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/home/agilex/isaac/isaacsim}"
export PYTHONEXE="$INSTALL_DIR/venv/bin/python"
export PYTHONPATH="$INSTALL_DIR/venv/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
# USD is also needed by standalone asset tools/tests before SimulationApp starts.
usd_libraries=("$ISAAC_SIM_PATH"/extscache/omni.usd.libs-*)
if [[ -d "${usd_libraries[0]}" ]]; then
  export PYTHONPATH="${usd_libraries[0]}:$PYTHONPATH"
  export LD_LIBRARY_PATH="${usd_libraries[0]}/bin:${usd_libraries[0]}/deps${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PATH="$INSTALL_DIR/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$INSTALL_DIR/pip-cache"
export HF_HOME="$INSTALL_DIR/huggingface"
export TORCH_EXTENSIONS_DIR="$INSTALL_DIR/torch_extensions"
export MPLCONFIGDIR="$INSTALL_DIR/matplotlib"
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
unset CONDA_PREFIX CONDA_DEFAULT_ENV
exec "$ISAAC_SIM_PATH/python.sh" "$@"
