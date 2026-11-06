#!/usr/bin/env bash
# 多实例并行生成数据：起 N 个互相独立的 Isaac 进程，各自不同随机种子、各写各的目录。
# 每个实例都是完整的 load -> plan -> render -> store 流水线，互不干扰，单实例崩了不影响其它。
#
#   bash scripts/run_parallel.sh                          # 默认 3 实例 × 每实例 5 条
#   INSTANCES=2 PER_INSTANCE=10 bash scripts/run_parallel.sh
#   TASK_CFG=local_configs/xxx.yaml INSTANCES=3 bash scripts/run_parallel.sh
#   WATCH_VRAM=0 bash scripts/run_parallel.sh             # 关掉显存采样
#
# 显存峰值会写进 $OUT_ROOT/gpu_watch.csv，跑完自动汇总并给出下一档建议。
set -uo pipefail

INSTANCES="${INSTANCES:-3}"           # 并行实例数
PER_INSTANCE="${PER_INSTANCE:-5}"     # 每个实例采样多少次
BASE_SEED="${BASE_SEED:-42}"          # 第 i 个实例用 BASE_SEED+i，保证数据不重复
OUT_ROOT="${OUT_ROOT:-output/parallel}"
TASK_CFG="${TASK_CFG:-local_configs/custom_pick_bread.yaml}"
WATCH_VRAM="${WATCH_VRAM:-1}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p "$OUT_ROOT"

watch_pid=""
if [ "$WATCH_VRAM" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits \
    -l 5 > "$OUT_ROOT/gpu_watch.csv" 2>/dev/null &
  watch_pid=$!
  echo "[vram] 每 5 秒采样一次显存 -> $OUT_ROOT/gpu_watch.csv"
fi

pids=()
names=()
for i in $(seq 1 "$INSTANCES"); do
  seed=$((BASE_SEED + i))
  out="$OUT_ROOT/inst_${i}/"
  log="$OUT_ROOT/inst_${i}.log"
  echo "[launch] inst_${i}: seed=${seed} 采样=${PER_INSTANCE} 输出=${out}"
  TASK_CFG="$TASK_CFG" OUT_DIR="$out" SEED="$seed" \
    bash scripts/run_custom_scene.sh "$PER_INSTANCE" > "$log" 2>&1 &
  pids+=($!)
  names+=("inst_${i}")
done

cleanup() {
  echo
  echo "[stop] 收到中断，终止全部实例"
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  [ -n "$watch_pid" ] && kill "$watch_pid" 2>/dev/null
  exit 130
}
trap cleanup INT TERM

echo "[wait] ${#pids[@]} 个实例并发运行中，Ctrl+C 会全部终止"
fail=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "[done] ${names[$idx]} 正常退出"
  else
    echo "[fail] ${names[$idx]} 非零退出，看 $OUT_ROOT/${names[$idx]}.log"
    fail=$((fail + 1))
  fi
done

if [ -n "$watch_pid" ]; then
  kill "$watch_pid" 2>/dev/null
  wait "$watch_pid" 2>/dev/null
fi

echo
echo "================ 汇总 ================"
ok_sum=0
for i in $(seq 1 "$INSTANCES"); do
  log="$OUT_ROOT/inst_${i}.log"
  if [ ! -s "$log" ]; then
    printf "  inst_%-3s 无日志（进程没起来）\n" "$i"
    continue
  fi
  # stdout 里每个成功 episode 会打一行 "Task is successful"；成功率汇总行只在 profile 日志里，stdout 查不到
  ok=$(grep -c "Task is successful" "$log" 2>/dev/null)
  err=$(grep -c "Traceback\|CUDA out of memory\|CUDA_ERROR_OUT_OF_MEMORY" "$log" 2>/dev/null)
  ok_sum=$((ok_sum + ok))
  printf "  inst_%-3s %s/%s 成功   异常行:%s\n" "$i" "$ok" "$PER_INSTANCE" "$err"
done
echo "  ------------------------------------"
printf "  合计      %s/%s\n" "$ok_sum" "$((INSTANCES * PER_INSTANCE))"
printf "  进程异常  %s 个\n" "$fail"

if [ -s "$OUT_ROOT/gpu_watch.csv" ]; then
  peak=$(awk -F', *' '{v=$2+0; if (v>m) m=v} END{printf "%.0f", m}' "$OUT_ROOT/gpu_watch.csv")
  if [ -n "$peak" ] && [ "$peak" -gt 0 ]; then
    per=$((peak / INSTANCES))
    room=$(( (24564 - 4096 - peak) / per ))
    echo
    echo "================ 显存 ================"
    echo "  峰值 ${peak} MiB / 24564 MiB"
    echo "  单实例均摊约 ${per} MiB"
    if [ "$room" -gt 0 ]; then
      echo "  留 4 GB 余量还能加 ${room} 个实例（总共 $((INSTANCES + room)) 个）"
    else
      echo "  已到当前显存上限，不要再加实例"
    fi
  fi
fi
echo
echo "数据目录：$OUT_ROOT/inst_*/"
