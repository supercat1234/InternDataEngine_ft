#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
# TOTAL 为每实例尝试次数；每路 seed 固定为 BASE_SEED + 实例编号 - 1。
exec "${PI05_PYTHON:-/home/agilex/miniforge3/envs/lerobot/bin/python}" \
  scripts/eval/batch_cup_plate_pi05.py \
  --workers "${INSTANCES:-3}" --total "${TOTAL:-100}" --base-seed "${BASE_SEED:-42}" \
  --episode-timeout "${EPISODE_TIMEOUT:-600}" "$@"
