#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "${PI05_PYTHON:-/home/agilex/miniforge3/envs/lerobot/bin/python}" "$REPO_DIR/scripts/eval/serve_pi05.py" "$@"
