#!/bin/bash
# 简化版课程学习训练脚本
#
# 2 个 stage，每 50 步做一次 refilter：
#   Stage 1: reward ∈ (0, 3)   的样本
#   Stage 2: reward ∈ (0, 1.5) 的样本（更难）
#
# Stage 2 直接从 Stage 1 的 FSDP checkpoint 恢复（通过 patch_checkpoint_for_stage.py），
# 保留 optimizer state（Adam m₁/m₂ + LR scheduler），不需要导出 HF 格式。
#
# 用法:
#   bash scripts/run_curriculum_training.sh --raw-data data/xxx [--output-dir DIR] [--dry-run]

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

[ -f .env ] && . .env

# ── 环境变量 ──────────────────────────────────────────────────────────────────
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/cache/triton}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
export TMPDIR="${TMPDIR:-/tmp}"
export TESTBED_ROOT="${TESTBED_ROOT:-/tmp/testbed}"
export VLLM_FLASH_ATTN_VERSION=2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export RAY_worker_register_timeout_seconds=600

MODEL="${BASE_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
CKPT_BASE="${CKPT_BASE:-ckpts}"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 8)
HALF_NUM_GPUS=$((NUM_GPUS > 0 ? NUM_GPUS / 2 : 4))
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-$HALF_NUM_GPUS}"
NUM_TRAINING_ENGINES="${NUM_TRAINING_ENGINES:-$HALF_NUM_GPUS}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"

TRAIN_REWARD_CONFIG="configs/rewards/baseline_4b.yaml"
PREFILTER_N_SAMPLES=4
PREFILTER_DP_SIZE=8
REFILTER_INTERVAL=50

# ── 固定课程配置 ──────────────────────────────────────────────────────────────
# Stage 1: 简单样本 reward ∈ (0, 3)
# Stage 2: 难样本   reward ∈ (0, 2)
# Stage 
STAGE_MAX_REWARDS=(3.0 1.5)
STEPS_PER_STAGE=100

# ── 参数解析（只保留必要的）─────────────────────────────────────────────────
RAW_DATA=""
OUTPUT_DIR="$REPO_ROOT/output/curriculum_simple"
DRY_RUN=false
RUN_PREFIX="curriculum_2stg"

while [[ $# -gt 0 ]]; do
  case $1 in
    --raw-data)    RAW_DATA="$2";    shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
    --run-prefix)  RUN_PREFIX="$2";  shift 2 ;;
    --dry-run)     DRY_RUN=true;     shift   ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

[ -z "$RAW_DATA" ] && { echo "ERROR: --raw-data is required"; exit 1; }
[[ "$RAW_DATA" != /* ]] && RAW_DATA="$REPO_ROOT/$RAW_DATA"
[ "$DRY_RUN" = false ] && [ ! -f "$RAW_DATA/train.parquet" ] && \
  { echo "ERROR: train.parquet not found in $RAW_DATA"; exit 1; }

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Simplified Curriculum Training (2 stages, refilter@${REFILTER_INTERVAL}steps)"
echo "  Stage 1: reward ∈ (0, ${STAGE_MAX_REWARDS[0]})"
echo "  Stage 2: reward ∈ (0, ${STAGE_MAX_REWARDS[1]})"
echo "  Steps/stage : $STEPS_PER_STAGE"
echo "  Raw data    : $RAW_DATA"
echo "  Output dir  : $OUTPUT_DIR"
echo "============================================================"

# ── 工具函数 ──────────────────────────────────────────────────────────────────
cleanup() {
  uv run ray stop 2>/dev/null || true
  pkill -9 VLLM 2>/dev/null || true
  pkill -9 uv 2>/dev/null || true
  uv cache prune 2>/dev/null || true
  sleep 2
}

run_cmd() {
  echo ">>> $*"
  [ "$DRY_RUN" = true ] && { echo "[DRY RUN] skipped"; return 0; }
  "$@"
}

# ── 训练函数 ──────────────────────────────────────────────────────────────────
run_train() {
  local _run_name="$1"
  local _data_dir="$2"
  local _ckpt_path="$3"
  local _resume_mode="${4:-none}"
  local _resume_path="${5:-}"

  [[ "$_data_dir" != /* ]] && _data_dir="$REPO_ROOT/$_data_dir"

  local _resume_args=("trainer.resume_mode=${_resume_mode}")
  [ -n "$_resume_path" ] && _resume_args+=("trainer.resume_path=${_resume_path}")

  uv run --isolated -m src.train \
    +run_async_trainer=true \
    "data.train_data=['${_data_dir}/train.parquet']" \
    "data.val_data=['${_data_dir}/validation.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.algorithm.grpo_norm_by_std=false \
    trainer.algorithm.policy_loss_type="gspo" \
    trainer.algorithm.eps_clip_low=3e-4 \
    trainer.algorithm.eps_clip_high=4e-4 \
    trainer.algorithm.loss_reduction="sequence_mean" \
    trainer.algorithm.use_kl_loss=False \
    trainer.algorithm.use_kl_in_reward=False \
    trainer.policy.model.path="${MODEL}" \
    trainer.placement.colocate_all=false \
    trainer.placement.colocate_policy_ref=true \
    trainer.strategy=fsdp2 \
    trainer.policy.fsdp_config.cpu_offload=true \
    trainer.policy.fsdp_config.reshard_after_forward=true \
    trainer.policy.fsdp_config.fsdp_size=-1 \
    trainer.fully_async.num_parallel_generation_workers="${BATCH_SIZE}" \
    trainer.placement.policy_num_gpus_per_node="${NUM_TRAINING_ENGINES}" \
    trainer.placement.ref_num_gpus_per_node="${NUM_TRAINING_ENGINES}" \
    trainer.placement.policy_num_nodes=1 \
    trainer.placement.ref_num_nodes=1 \
    trainer.policy.sequence_parallel_size=1 \
    generator.num_inference_engines="${NUM_INFERENCE_ENGINES}" \
    generator.inference_engine_tensor_parallel_size=1 \
    +generator.traj_dir="${_ckpt_path}trajectories/" \
    +generator.exp_config="${TRAIN_REWARD_CONFIG}" \
    +generator.prompts.system_prompt="templates/system_prompt_custom_finish.j2" \
    +generator.prompts.user_prompt="templates/file_module_custom_finish.j2" \
    +generator.engine_init_kwargs.enable_auto_tool_choice=true \
    +generator.engine_init_kwargs.tool_call_parser=hermes \
    +generator.engine_init_kwargs.max_model_len=40960 \
    +generator.engine_init_kwargs.disable_cascade_attn=true \
    generator.eval_n_samples_per_prompt=1 \
    trainer.epochs=1 \
    trainer.eval_batch_size=32 \
    trainer.eval_before_train=true \
    trainer.eval_interval=-1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size="${BATCH_SIZE}" \
    trainer.policy_mini_batch_size="${BATCH_SIZE}" \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    trainer.dump_data_batch=true \
    trainer.export_path="${_ckpt_path}exported_model/" \
    trainer.hf_save_interval=25 \
    trainer.ckpt_interval="${REFILTER_INTERVAL}" \
    trainer.use_sample_packing=false \
    trainer.max_prompt_length=40960 \
    generator.sampling_params.max_generate_length="${MAX_LENGTH}" \
    generator.sampling_params.temperature=1.0 \
    generator.max_input_length=40960 \
    generator.max_num_batched_tokens=131072 \
    generator.max_turns=10 \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    generator.backend=vllm \
    generator.run_engines_locally=True \
    generator.enable_http_endpoint=True \
    generator.http_endpoint_host='0.0.0.0' \
    generator.http_endpoint_port=8080 \
    generator.weight_sync_backend=nccl \
    generator.async_engine=true \
    generator.batched=false \
    generator.n_samples_per_prompt="${N_ROLLOUTS}" \
    generator.gpu_memory_utilization=0.8 \
    generator.enforce_eager=false \
    +trainer.step_wise_training=false \
    trainer.logger="wandb" \
    trainer.project_name="soni_curriculum_4b" \
    trainer.run_name="${_run_name}" \
    "${_resume_args[@]}" \
    trainer.ckpt_path="${_ckpt_path}" \
    trainer.max_ckpts_to_keep=5
}

# ── refilter: 用当前 ckpt 重新 rollout 打分 ──────────────────────────────────
run_refilter() {
  local _stage="$1"
  local _ckpt_dir="$2"
  local _output="$3"
  local _max_reward="$4"
  local _target_samples=$((STEPS_PER_STAGE * BATCH_SIZE))

  cleanup
  run_cmd uv run python "$SCRIPT_DIR/prefilter_data.py" \
    --model 4b \
    --checkpoint "$_ckpt_dir" \
    --input "$RAW_DATA/train.parquet" \
    --output "$_output" \
    --config configs/rewards/prefilter_localization_only.yaml \
    --n-samples "$PREFILTER_N_SAMPLES" \
    --dp-size "$PREFILTER_DP_SIZE" \
    --min-reward 0.0 \
    --max-reward "$_max_reward" \
    --shuffle \
    --target-valid-samples "$_target_samples" \
    --resume \
    2>&1 | tee "$LOG_DIR/refilter_stage${_stage}.log"
}

# ── 准备 stage 数据 ──────────────────────────────────────────────────────────
prepare_stage_data() {
  local _reward_data="$1"
  local _stage_dir="$2"
  local _max_reward="$3"
  local _max_samples=$((STEPS_PER_STAGE * BATCH_SIZE))

  mkdir -p "$_stage_dir"
  run_cmd uv run python "$SCRIPT_DIR/select_curriculum_stage.py" \
    --input "$_reward_data" \
    --min-reward 0.0 \
    --max-reward "$_max_reward" \
    --max-samples "$_max_samples" \
    --output "$_stage_dir/train.parquet"

  # validation set
  [ -f "$RAW_DATA/validation.parquet" ] && cp "$RAW_DATA/validation.parquet" "$_stage_dir/validation.parquet"
}

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: 初始 prefilter（base model）+ 训练
# ══════════════════════════════════════════════════════════════════════════════
STAGE1_MAX_REWARD="${STAGE_MAX_REWARDS[0]}"
STAGE1_RUN="${RUN_PREFIX}_s1"
STAGE1_CKPT="${CKPT_BASE}/${STAGE1_RUN}/"
STAGE1_DATA="$OUTPUT_DIR/stage1_data"
PREFILTER_DIR="$OUTPUT_DIR/prefilter"

echo ""
echo "=== Stage 1: prefilter with base model, reward ∈ (0, ${STAGE1_MAX_REWARD}) ==="

# 初始 prefilter（用 base model rollout）
if [ "$DRY_RUN" = false ] && [ -f "$PREFILTER_DIR/train.parquet" ]; then
  echo "--- Prefilter already exists, skipping ---"
else
  cleanup
  TARGET_VALID=$((STEPS_PER_STAGE * BATCH_SIZE))
  run_cmd uv run python "$SCRIPT_DIR/prefilter_data.py" \
    --model 4b \
    --input "$RAW_DATA/train.parquet" \
    --output "$PREFILTER_DIR" \
    --config configs/rewards/prefilter_localization_only.yaml \
    --n-samples "$PREFILTER_N_SAMPLES" \
    --dp-size "$PREFILTER_DP_SIZE" \
    --shuffle \
    --min-reward 0.0 \
    --max-reward "$STAGE1_MAX_REWARD" \
    --target-valid-samples "$TARGET_VALID" \
    --resume \
    2>&1 | tee "$LOG_DIR/prefilter_base.log"
fi

# 选样本 + 训练
prepare_stage_data "$PREFILTER_DIR/train.parquet" "$STAGE1_DATA" "$STAGE1_MAX_REWARD"

echo "--- Training Stage 1 ($STEPS_PER_STAGE steps) ---"
cleanup
mkdir -p "$STAGE1_CKPT"
run_train "$STAGE1_RUN" "$STAGE1_DATA" "$STAGE1_CKPT" "none" "" \
  2>&1 | tee "$LOG_DIR/train_stage1.log"

# ══════════════════════════════════════════════════════════════════════════════
# Refilter: 用 Stage 1 ckpt 重新评估难度
# ══════════════════════════════════════════════════════════════════════════════
REFILTER_DIR="$OUTPUT_DIR/refilter_s1"
echo ""
echo "=== Refilter: re-evaluate with Stage 1 checkpoint ==="
run_refilter 1 "$STAGE1_CKPT" "$REFILTER_DIR" "${STAGE_MAX_REWARDS[1]}"

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: 从 Stage 1 FSDP ckpt 恢复 + 更难的样本
# ══════════════════════════════════════════════════════════════════════════════
STAGE2_MAX_REWARD="${STAGE_MAX_REWARDS[1]}"
STAGE2_RUN="${RUN_PREFIX}_s2"
STAGE2_CKPT="${CKPT_BASE}/${STAGE2_RUN}/"
STAGE2_DATA="$OUTPUT_DIR/stage2_data"
STAGE2_INIT_CKPT="${CKPT_BASE}/${STAGE2_RUN}_init/"

echo ""
echo "=== Stage 2: reward ∈ (0, ${STAGE2_MAX_REWARD}), resume from Stage 1 FSDP ckpt ==="

# 选样本
REFILTER_TRAIN="$REFILTER_DIR/train.parquet"
if [ "$DRY_RUN" = false ] && [ ! -f "$REFILTER_TRAIN" ]; then
  echo "WARNING: refilter failed, falling back to initial prefilter data"
  REFILTER_TRAIN="$PREFILTER_DIR/train.parquet"
fi
prepare_stage_data "$REFILTER_TRAIN" "$STAGE2_DATA" "$STAGE2_MAX_REWARD"

# Checkpoint surgery: 复制 Stage 1 的 FSDP ckpt（含 optimizer state）到 Stage 2 的 global_step_0
STAGE2_RESUME_MODE="none"
STAGE2_RESUME_PATH=""

if [ "$DRY_RUN" = true ]; then
  echo "[DRY RUN] checkpoint surgery: $STAGE1_CKPT → ${STAGE2_INIT_CKPT}global_step_0"
  STAGE2_RESUME_MODE="from_path"
  STAGE2_RESUME_PATH="${STAGE2_INIT_CKPT}global_step_0"
elif [ -d "$STAGE1_CKPT" ]; then
  echo "--- Checkpoint surgery: Stage 1 FSDP ckpt → Stage 2 init ---"
  uv run python "$SCRIPT_DIR/patch_checkpoint_for_stage.py" \
    --src-ckpt-dir "$STAGE1_CKPT" \
    --dst-ckpt-dir "$STAGE2_INIT_CKPT" \
    2>&1 | tee "$LOG_DIR/patch_ckpt_s2.log"

  if [ -d "${STAGE2_INIT_CKPT}global_step_0" ]; then
    STAGE2_RESUME_MODE="from_path"
    STAGE2_RESUME_PATH="${STAGE2_INIT_CKPT}global_step_0"
    echo "  Resume from: $STAGE2_RESUME_PATH"
  else
    echo "WARNING: patch failed, Stage 2 will start from base model"
  fi
else
  echo "WARNING: Stage 1 ckpt not found at $STAGE1_CKPT, Stage 2 starts from base model"
fi

# 训练
echo "--- Training Stage 2 ($STEPS_PER_STAGE steps) ---"
cleanup
mkdir -p "$STAGE2_CKPT"
run_train "$STAGE2_RUN" "$STAGE2_DATA" "$STAGE2_CKPT" "$STAGE2_RESUME_MODE" "$STAGE2_RESUME_PATH" \
  2>&1 | tee "$LOG_DIR/train_stage2.log"

# ── 完成 ──────────────────────────────────────────────────────────────────────
cleanup
echo ""
echo "============================================================"
echo "Curriculum training complete"
echo "  Stage 1 ckpt: $STAGE1_CKPT"
echo "  Stage 2 ckpt: $STAGE2_CKPT"
echo "  Logs: $LOG_DIR"
echo "============================================================"
