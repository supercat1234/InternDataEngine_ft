#!/usr/bin/env bash
# 跑自定义场景：用法同 run-example.sh
#   bash scripts/run_custom_scene.sh 1
#   bash scripts/run_custom_scene.sh 1 --load_stage.scene_loader.args.simulator.headless=false
set -euo pipefail

INSTALL_DIR="/home/agilex/qhx/.interndata"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASK_CFG="${TASK_CFG:-local_configs/custom_pick_bread.yaml}"
OUT_DIR="${OUT_DIR:-output/custom_pick_bread/}"
CONFIG="${CONFIG:-configs/simbox/de_plan_with_render_template.yaml}"  # 换 pipe 并行模式时指到 de_pipe_custom_pick.yaml
SEED="${SEED:-42}"                    # 多实例并行时必须给不同值，否则几个实例生成一模一样的数据
DEBUG="${DEBUG:-1}"                   # 1=出错立刻抛完整堆栈；批量生产可以设 0
NUM="${1:-1}"

cd "$REPO_DIR"

debug_flags=()
[ "$DEBUG" = "1" ] && debug_flags=(--debug)

# pipe 配置里还有第二个 scene_loader（dedump 阶段的），任务路径必须同步，否则规划阶段加载 A、渲染阶段加载 B
pipe_flags=()
case "$CONFIG" in
  *pipe*) pipe_flags=("--dedump_stage.scene_loader.args.cfg_path=$TASK_CFG") ;;
esac

exec "$INSTALL_DIR/python.sh" "$INSTALL_DIR/launch.py" \
  --config "$CONFIG" \
  "${debug_flags[@]}" --random_seed="$SEED" \
  "${pipe_flags[@]}" \
  --load_stage.scene_loader.args.cfg_path="$TASK_CFG" \
  --load_stage.layout_random_generator.args.random_num="$NUM" \
  --load_stage.layout_random_generator.args.strict_mode=false \
  --store_stage.writer.args.batch_async=false \
  --store_stage.writer.args.output_dir="$OUT_DIR" \
  "${@:2}"
