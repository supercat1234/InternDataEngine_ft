#!/usr/bin/env bash
# Independent cup-to-plate workers; each owns a process group, seed and output directory.
set -uo pipefail

usage() {
  cat <<'EOF'
用法：bash scripts/run_cup_plate_parallel.sh [--dry-run] [单实例配置覆盖参数...]
环境变量：
  INSTANCES=2       并行实例数
  PER_INSTANCE=5    每实例布局数；strict为成功目标，否则为尝试数
  BASE_SEED=42      第一个实例的种子；后续依次加一
  OUT_ROOT=...      输出根目录；默认 output/cup_plate_desk/v0.7/parallel/时间_进程号
  WATCH_VRAM=1     每5秒记录各GPU的显存和利用率；0关闭
  LAUNCH_DELAY=3   启动相邻实例的间隔秒数；0立即启动
  STRICT_MODE=false 是否失败后重新采样直到达到成功目标
  MAX_RETRIES=-1   strict每条目标样本额外重试上限；-1不限次数

例：INSTANCES=2 PER_INSTANCE=10 bash scripts/run_cup_plate_parallel.sh
    bash scripts/run_cup_plate_parallel.sh --dry-run
    bash scripts/run_cup_plate_parallel.sh --load_stage.scene_loader.args.simulator.headless=false
EOF
}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1
INSTANCES="${INSTANCES:-2}"
PER_INSTANCE="${PER_INSTANCE:-5}"
BASE_SEED="${BASE_SEED:-42}"
OUT_ROOT="${OUT_ROOT:-output/cup_plate_desk/v0.7/parallel/$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "$OUT_ROOT" != /* ]]; then OUT_ROOT="$REPO_DIR/$OUT_ROOT"; fi
WATCH_VRAM="${WATCH_VRAM:-1}"
LAUNCH_DELAY="${LAUNCH_DELAY:-3}"
dry_run=0
overrides=()
for arg in "$@"; do
  case "$arg" in
    --help|-h) usage; exit 0 ;;
    --dry-run) dry_run=1 ;;
    --name|--name=*|--random_seed|--random_seed=*|--store_stage.writer.args.output_dir|--store_stage.writer.args.output_dir=*|--load_stage.layout_random_generator.args.random_num|--load_stage.layout_random_generator.args.random_num=*)
      echo "运行名称由脚本分配；请用 BASE_SEED、OUT_ROOT、PER_INSTANCE 设置种子、目录和尝试次数。" >&2
      exit 2 ;;
    *) overrides+=("$arg") ;;
  esac
done
for value in "$INSTANCES" "$PER_INSTANCE"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "INSTANCES 和 PER_INSTANCE 必须为正整数。" >&2; exit 2
  fi
done
if [[ ! "$BASE_SEED" =~ ^(0|[1-9][0-9]*)$ || ${#BASE_SEED} -gt 10 || ${#INSTANCES} -gt 10 ]] || (( BASE_SEED > 4294967295 || INSTANCES - 1 > 4294967295 - BASE_SEED )); then
  echo "所有实例的种子必须在 0..4294967295 内。" >&2; exit 2
fi
if [[ ! "$LAUNCH_DELAY" =~ ^(0|[1-9][0-9]*)$ ]] || [[ ! "$WATCH_VRAM" =~ ^[01]$ ]]; then
  echo "LAUNCH_DELAY 必须为非负整数，WATCH_VRAM 必须为0或1。" >&2; exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "缺少 setsid（util-linux），无法保证中断时清理全部子进程。" >&2; exit 2
fi

if (( dry_run )); then
  for ((i=1; i<=INSTANCES; i++)); do
    printf '[dry-run] STRICT_MODE=%q MAX_RETRIES=%q SEED=%q OUT_DIR=%q bash %q %q' \
      "${STRICT_MODE:-false}" "${MAX_RETRIES:--1}" "$((BASE_SEED+i-1))" "$OUT_ROOT/inst_$i/" \
      "$REPO_DIR/scripts/run_cup_plate_desk.sh" "$PER_INSTANCE"
    printf ' %q' "--name=$OUT_ROOT/inst_$i/engine"
    if (( ${#overrides[@]} )); then printf ' %q' "${overrides[@]}"; fi
    printf '\n'
  done
  exit 0
fi

# Refuse to overwrite logs or mix another run's data into these instance directories.
for ((i=1; i<=INSTANCES; i++)); do
  if [[ -e "$OUT_ROOT/inst_$i" || -e "$OUT_ROOT/inst_$i.log" ]]; then
    echo "输出目录已有实例数据或日志，请更换 OUT_ROOT：$OUT_ROOT" >&2; exit 2
  fi
done
if [[ -e "$OUT_ROOT/summary.csv" || -e "$OUT_ROOT/instances.tsv" || -e "$OUT_ROOT/gpu_watch.csv" ]]; then
  echo "OUT_ROOT 已包含运行记录，请使用新的目录：$OUT_ROOT" >&2; exit 2
fi
mkdir -p "$OUT_ROOT" || exit 1

pids=()
statuses=()
watch_pid=""
stop_watch() {
  if [[ -n "$watch_pid" ]]; then
    kill -TERM -- "-$watch_pid" 2>/dev/null || true
    wait "$watch_pid" 2>/dev/null || true
    watch_pid=""
  fi
}
interrupt() {
  local code="$1" timer_pid
  trap '' INT TERM
  echo "[stop] 正在终止本次启动的全部实例及其子进程……" >&2
  stop_watch
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  # Isaac may ignore TERM during CUDA work; bound shutdown time to five seconds.
  (sleep 5; for pid in "${pids[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done) &
  timer_pid=$!
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  # Keep the timeout even after launcher shells exit: their Python children
  # can still be alive in the worker's process group.
  wait "$timer_pid" 2>/dev/null || true
  exit "$code"
}
trap 'interrupt 130' INT
trap 'interrupt 143' TERM
trap stop_watch EXIT

if [[ "$WATCH_VRAM" == 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  printf 'timestamp,gpu_index,memory_total_MiB,memory_used_MiB,utilization_percent\n' > "$OUT_ROOT/gpu_watch.csv"
  setsid nvidia-smi --query-gpu=timestamp,index,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -l 5 >> "$OUT_ROOT/gpu_watch.csv" 2> "$OUT_ROOT/gpu_watch.stderr" &
  watch_pid=$!
fi

printf 'instance\tpid\tseed\trequested_layouts\toutput\tlog\n' > "$OUT_ROOT/instances.tsv"
for ((i=1; i<=INSTANCES; i++)); do
  seed=$((BASE_SEED+i-1))
  out="$OUT_ROOT/inst_$i/"
  log="$OUT_ROOT/inst_$i.log"
  echo "[launch] inst_$i seed=$seed 布局数参数=$PER_INSTANCE 输出=$out"
  SEED="$seed" OUT_DIR="$out" setsid bash "$REPO_DIR/scripts/run_cup_plate_desk.sh" \
    "$PER_INSTANCE" "--name=${out%/}/engine" "${overrides[@]}" > "$log" 2>&1 &
  pids+=("$!")
  printf 'inst_%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "${pids[-1]}" "$seed" "$PER_INSTANCE" "$out" "$log" >> "$OUT_ROOT/instances.tsv"
  if (( i < INSTANCES && LAUNCH_DELAY > 0 )); then sleep "$LAUNCH_DELAY"; fi
done

fail=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then code=0; else code=$?; fail=$((fail+1)); fi
  statuses+=("$code")
  echo "[done] inst_$((idx+1)) exit=$code"
done
stop_watch

printf 'instance,seed,requested_layouts,task_successes,saved_episodes,exit_code\n' > "$OUT_ROOT/summary.csv"
success_total=0
saved_total=0
for ((i=1; i<=INSTANCES; i++)); do
  log="$OUT_ROOT/inst_$i.log"
  # Count both successful execution and actual saved episodes: writes can fail
  # after a task has already printed success. Only count inside this worker.
  success=$(awk '/^[[:space:]]*Task is successful[[:space:]]*$/ {n++} END {print n+0}' "$log")
  saved=0
  if [[ -d "$OUT_ROOT/inst_$i" ]]; then
    while IFS= read -r -d '' metadata; do saved=$((saved+1)); done < <(find "$OUT_ROOT/inst_$i" -type f -name meta_info.pkl -print0)
  fi
  success_total=$((success_total+success))
  saved_total=$((saved_total+saved))
  printf 'inst_%s,%s,%s,%s,%s,%s\n' "$i" "$((BASE_SEED+i-1))" "$PER_INSTANCE" "$success" "$saved" "${statuses[$((i-1))]}" >> "$OUT_ROOT/summary.csv"
  echo "[summary] inst_$i 成功=$success/$PER_INSTANCE 已保存=$saved exit=${statuses[$((i-1))]}"
done
echo "[total] 配置布局数=$((INSTANCES*PER_INSTANCE)) 成功=$success_total 已保存=$saved_total 异常进程=$fail"
if [[ -s "$OUT_ROOT/gpu_watch.csv" ]]; then
  awk -F', *' 'NR>1 && $2 ~ /^[0-9]+$/ {gpu=$2; total[gpu]=$3; if ($4>peak[gpu]) peak[gpu]=$4}
    END {for (gpu in total) printf "[vram] GPU %s 峰值 %.0f / %.0f MiB（含其他进程）\n",gpu,peak[gpu],total[gpu]}' "$OUT_ROOT/gpu_watch.csv"
fi
echo "数据和日志：$OUT_ROOT"
if (( fail > 0 )); then exit 1; fi
exit 0
