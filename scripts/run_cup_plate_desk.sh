#!/usr/bin/env bash
# First argument: attempted layouts, or successful layouts when STRICT_MODE=true.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
NUM="${1:-1}"
if [[ ! "$NUM" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash scripts/run_cup_plate_desk.sh [positive layout count] [config overrides...]" >&2
  exit 2
fi
if [[ $# -gt 0 ]]; then shift; fi
exec bash "$REPO_DIR/scripts/isaac_python.sh" "$REPO_DIR/scripts/launch_isaac.py" \
  --config configs/simbox/de_plan_with_render_template.yaml \
  --debug --random_seed="${SEED:-42}" \
  --load_stage.scene_loader.args.cfg_path=local_configs/cup_plate_desk/task.yaml \
  --load_stage.layout_random_generator.args.random_num="$NUM" \
  --load_stage.layout_random_generator.args.strict_mode="${STRICT_MODE:-false}" \
  --load_stage.layout_random_generator.args.max_retries="${MAX_RETRIES:--1}" \
  --store_stage.writer.args.batch_async=false \
  --store_stage.writer.args.output_dir="${OUT_DIR:-output/cup_plate_desk/v0.7/}" \
  "$@"
