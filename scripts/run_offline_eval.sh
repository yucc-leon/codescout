#!/bin/bash
# 离线评估入口：在 validation.parquet 上跑 rollout，不训练。
# 使用 prefilter_data.py --rollout-only，数据集为现成 validation.parquet。
#
# Usage:
#   bash scripts/run_offline_eval.sh
#   DATA_PATH=/path/to/data bash scripts/run_offline_eval.sh
#   CHECKPOINT=/path/to/ckpt bash scripts/run_offline_eval.sh --max-samples 50
#
# 环境变量:
#   DATA_PATH       数据目录，需含 validation.parquet
#   VALIDATION_FILE 直接指定 validation 文件路径（覆盖 DATA_PATH）
#   CHECKPOINT      可选，评估用模型/ckpt；不设则用 base 4b
#   MODEL           base 模型 4b/14b（未设 CHECKPOINT 时）
#   OUTPUT_DIR      结果目录
#   CONFIG          实验配置 yaml
#   MAX_SAMPLES     最多评估条数
#   N_SAMPLES       每条 rollout 次数取最优
#   RESUME          1 则从已有 rollout_results.jsonl 续跑

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
[ -f .env ] && . .env

DATA_PATH="${DATA_PATH:-data/adityasoni17__SWE-smith-py-code-search_train}"
[[ "$DATA_PATH" != /* ]] && DATA_PATH="$REPO_ROOT/$DATA_PATH"
if [ -n "${VALIDATION_FILE:-}" ]; then
  VAL_PARQUET="$VALIDATION_FILE"
else
  VAL_PARQUET="${DATA_PATH}/validation.parquet"
fi

if [ ! -f "$VAL_PARQUET" ]; then
  echo "Error: validation not found: $VAL_PARQUET"
  echo "Set DATA_PATH or VALIDATION_FILE."
  exit 1
fi

CHECKPOINT="${CHECKPOINT:-}"
MODEL="${MODEL:-4b}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval_output/offline_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/rewards/baseline_4b.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
N_SAMPLES="${N_SAMPLES:-1}"
RESUME="${RESUME:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --output|-o)   OUTPUT_DIR="$2"; shift 2 ;;
    --checkpoint|-c) CHECKPOINT="$2"; shift 2 ;;
    --config)      CONFIG="$2"; shift 2 ;;
    --resume|-r)   RESUME=1; shift ;;
    --help|-h)
      echo "Usage: $0 [--max-samples N] [--output DIR] [--checkpoint P] [--config PATH] [--resume]"
      exit 0 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/eval.log"

echo "========== Offline Eval =========="
echo "Validation: $VAL_PARQUET"
echo "Output:     $OUTPUT_DIR"
echo "Checkpoint: ${CHECKPOINT:-base $MODEL}"
echo "=================================="

RUN_ARGS=(--input "$VAL_PARQUET" --output "$OUTPUT_DIR" --config "$CONFIG" --rollout-only --n-samples "$N_SAMPLES")
[ -n "$CHECKPOINT" ] && RUN_ARGS+=(--checkpoint "$CHECKPOINT") || RUN_ARGS+=(--model "$MODEL")
[ -n "$MAX_SAMPLES" ] && RUN_ARGS+=(--max-samples "$MAX_SAMPLES")
[ "$RESUME" = "1" ] && RUN_ARGS+=(--resume)

uv run python scripts/prefilter_data.py "${RUN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

ROLLOUT_JSONL="${OUTPUT_DIR}/rollout_results.jsonl"
if [ -f "$ROLLOUT_JSONL" ]; then
  uv run python scripts/write_eval_summary.py "$ROLLOUT_JSONL" "$OUTPUT_DIR"
fi

echo "Done. Results: $OUTPUT_DIR (rollout_results.jsonl, eval_summary.json, eval.log)"
