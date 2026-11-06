#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
exec bash scripts/isaac_python.sh scripts/eval/rollout_cup_plate_pi05.py "$@"
