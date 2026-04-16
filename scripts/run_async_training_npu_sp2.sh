#!/bin/bash
# ============================================================================
# CodeScout RL training on Ascend NPU
#
# Adapted from run_async_training_4b.sh + run_ablation.sh patterns,
# with NPU-specific env vars from cann-recipes-train.
#
# Usage:
#   # Basic run
#   bash scripts/run_async_training_npu.sh \
#     -m /path/to/Qwen3-4B-Instruct-2507 \
#     -n 4 -b 4 -c 1 -i 4 -t 4 \
#     -d ./data/swe_smith/ \
#     -s /tmp/codescout_ckpts \
#     -r codescout-npu-4b
#
#   # Resume from checkpoint
#   bash scripts/run_async_training_npu.sh ... --resume
# ============================================================================

set -euo pipefail

# ==================== Parse arguments ====================
RESUME_MODE="none"
POSITIONAL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --resume) RESUME_MODE="auto" ;;
        *) POSITIONAL_ARGS+=("$arg") ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}"

while getopts ":m:n:d:s:l:o:i:t:b:c:r:w:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;  n ) N_ROLLOUTS=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;  s ) CKPT_PATH=$OPTARG;;
    l ) LCAL_PATH=$OPTARG;;  o ) OTHER_OPTION=$OPTARG;;
    i ) NUM_INFERENCE_ENGINES=$OPTARG;;  t ) NUM_TRAINING_ENGINES=$OPTARG;;
    b ) BATCH_SIZE=$OPTARG;;  c ) MICRO_BATCH_SIZE=$OPTARG;;
    r ) RUN_NAME=$OPTARG;;  w ) STEP_WISE=$OPTARG;;
  esac
done

# ==================== Ascend NPU environment ====================
# CANN runtime
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash 2>/dev/null || true

# Device visibility (8 NPUs)
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Ascend logging (quiet by default; set ASCEND_GLOBAL_LOG_LEVEL=0 for debug)
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0

# NPU memory allocator — reduce fragmentation for long-sequence training
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# HCCL (Ascend's collective communication library, replaces NCCL)
export HCCL_CONNECT_TIMEOUT=360
export HCCL_EXEC_TIMEOUT=360
export HCCL_IF_BASE_PORT=64033
export HCCL_OP_EXPANSION_MODE=AIV

# vLLM-Ascend specific
export VLLM_ASCEND_ENABLE_NZ=0

# General
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_worker_register_timeout_seconds=600
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
# Prevent triton namespace conflict in .pth auto-load (affects Ray workers)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export SKYRL_PYTHONPATH_EXPORT=1
export WANDB_INIT_TIMEOUT=300

# Repo cache for fast local clone (set by cluster entrypoint)
export REPO_CACHE="${REPO_CACHE:-}"

# Network proxy — preserve if already set (e.g. by cluster entrypoint).
# Git clone in rollout workers needs proxy to reach GitHub.
# vLLM HTTP endpoints (localhost) must NOT go through proxy.
if [ -n "${http_proxy:-}" ] || [ -n "${ALL_PROXY:-}" ]; then
    export no_proxy="localhost,127.0.0.1,0.0.0.0,$(hostname -i 2>/dev/null || echo '')"
    export NO_PROXY="$no_proxy"
    echo "Proxy preserved: ${http_proxy:-${ALL_PROXY:-}}, no_proxy=$no_proxy"
else
    # No proxy set — ensure clean state (dev machine / direct internet)
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY 2>/dev/null || true
fi
export SKYRL_LD_LIBRARY_PATH_EXPORT=1

# Cache dirs (from ablation script — avoid polluting home dir)
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/cache/triton}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
export TMPDIR="${TMPDIR:-/tmp}"
export TESTBED_ROOT="${TESTBED_ROOT:-/tmp/testbed}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$TESTBED_ROOT"

# ==================== Conda / Python ====================
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
CONDA_ENV="${CONDA_ENV:-$CONDA_BASE/envs/codescout-cann83}"
PYTHON="$CONDA_ENV/bin/python"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODESCOUT_ROOT="$(dirname "$SCRIPT_DIR")"
SKYRL_ROOT="$(dirname "$CODESCOUT_ROOT")/SkyRL"
export PYTHONPATH="$CODESCOUT_ROOT:$SKYRL_ROOT:${PYTHONPATH:-}"

# Source .env if present (wandb key, etc.)
[ -f "$CODESCOUT_ROOT/.env" ] && source "$CODESCOUT_ROOT/.env"

# ==================== Defaults ====================
MODEL_ALIAS=$(basename $MODEL)
NUM_GPUS=$($PYTHON -c "import torch_npu,torch;print(torch.npu.device_count())" 2>/dev/null || echo 8)
N_ROLLOUTS="${N_ROLLOUTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH=8192
RUN_NAME="${RUN_NAME:-${MODEL_ALIAS}-npu-${BATCH_SIZE}x${N_ROLLOUTS}}"
DATA_PATH="${DATA_PATH:-data/swe_smith}"
CKPT_PATH="${CKPT_PATH:-./ckpts/${MODEL_ALIAS}}"
LCAL_PATH="${LCAL_PATH:-$CKPT_PATH}"
HALF_NUM_GPUS=$((NUM_GPUS / 2))
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-$HALF_NUM_GPUS}"
NUM_TRAINING_ENGINES="${NUM_TRAINING_ENGINES:-$HALF_NUM_GPUS}"
STEP_WISE="${STEP_WISE:-false}"

mkdir -p "$CKPT_PATH" logs

# ==================== Resume / wandb tracking ====================
WANDB_ID_FILE="${CKPT_PATH}/wandb_run_id.txt"
WANDB_ARGS=""
ACTUAL_RESUME="$RESUME_MODE"

if [ "$RESUME_MODE" = "auto" ]; then
    latest_ckpt=$(ls -d "${CKPT_PATH}"/global_step_* 2>/dev/null | sort -t_ -k3 -rn | head -1 || true)
    if [ -n "$latest_ckpt" ] && [ -f "${latest_ckpt}/trainer_state.pt" ]; then
        step_num=$(basename "$latest_ckpt" | sed 's/global_step_//')
        echo "🔄 Resuming from checkpoint: step $step_num"
        if [ -f "$WANDB_ID_FILE" ]; then
            wandb_run_id=$(cat "$WANDB_ID_FILE")
            WANDB_ARGS="+trainer.wandb_run_id=${wandb_run_id}"
            echo "🔗 Resuming wandb run: $wandb_run_id"
        fi
        ACTUAL_RESUME="latest"
    else
        echo "⚠️  No valid checkpoint found, starting fresh"
        ACTUAL_RESUME="none"
    fi
fi

# ==================== Cleanup helper ====================
cleanup() {
    echo "🧹 Cleaning up..."
    ray stop 2>/dev/null || true
    sleep 3
}
trap cleanup EXIT

LOG_FILE="logs/$(date +%m%d_%H%M)_${RUN_NAME}.log"

# ==================== Launch ====================
echo "=========================================="
echo "📅 $(date '+%Y-%m-%d %H:%M:%S')"
echo "🚀 CodeScout NPU Training"
echo "📦 Model: $MODEL"
echo "📂 Data:  $DATA_PATH"
echo "💾 Ckpt:  $CKPT_PATH"
echo "🎯 Run:   $RUN_NAME"
echo "🔧 GPUs:  $NUM_GPUS (train=$NUM_TRAINING_ENGINES, infer=$NUM_INFERENCE_ENGINES)"
echo "🔄 Resume: $ACTUAL_RESUME"
echo "=========================================="

set -x

$PYTHON -m src.train_npu \
  +run_async_trainer=true \
  data.train_data="['$DATA_PATH/train.parquet']" \
  data.val_data="['$DATA_PATH/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.policy.model.path=${MODEL} \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.policy.fsdp_config.reshard_after_forward=true \
  trainer.policy.fsdp_config.fsdp_size=-1 \
  trainer.fully_async.num_parallel_generation_workers=${BATCH_SIZE} \
  trainer.placement.policy_num_gpus_per_node=${NUM_TRAINING_ENGINES} \
  trainer.placement.ref_num_gpus_per_node=${NUM_TRAINING_ENGINES} \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.policy.sequence_parallel_size=2 \
  generator.num_inference_engines=${NUM_INFERENCE_ENGINES} \
  generator.inference_engine_tensor_parallel_size=1 \
  +generator.traj_dir=${CKPT_PATH}/trajectories/ \
  +generator.engine_init_kwargs.enable_auto_tool_choice=true \
  +generator.engine_init_kwargs.tool_call_parser="hermes" \
  +generator.engine_init_kwargs.max_model_len=40960 \
  +generator.prompts.system_prompt="templates/system_prompt_custom_finish.j2" \
  +generator.prompts.user_prompt="templates/file_module_custom_finish.j2" \
  +generator.engine_init_kwargs.disable_cascade_attn=true \
  trainer.flash_attn=true \
  trainer.use_sample_packing=true \
  trainer.epochs=1 \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${BATCH_SIZE} \
  trainer.policy_mini_batch_size=${BATCH_SIZE} \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
  trainer.dump_data_batch=true \
  trainer.export_path="${CKPT_PATH}/exported_model/" \
  trainer.hf_save_interval=50 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=40960 \
  trainer.algorithm.policy_loss_type="gspo" \
  trainer.algorithm.eps_clip_low=0.0003 \
  trainer.algorithm.eps_clip_high=0.0004 \
  trainer.algorithm.loss_reduction="sequence_mean" \
  generator.sampling_params.max_generate_length=${MAX_LENGTH} \
  generator.sampling_params.temperature=1.0 \
  generator.max_input_length=40960 \
  generator.max_num_batched_tokens=131072 \
  generator.max_turns=10 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=False \
  trainer.algorithm.use_kl_in_reward=False \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.enable_http_endpoint=True \
  generator.http_endpoint_host='0.0.0.0' \
  generator.http_endpoint_port=8080 \
  generator.weight_sync_backend=hccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.n_samples_per_prompt=${N_ROLLOUTS} \
  generator.gpu_memory_utilization=0.85 \
  generator.enforce_eager=true \
  trainer.step_wise_training=${STEP_WISE} \
  trainer.logger="wandb" \
  trainer.project_name="code_search_npu" \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=${ACTUAL_RESUME} \
  trainer.ckpt_path="$LCAL_PATH" \
  trainer.max_ckpts_to_keep=5 \
  +generator.reward=configs/reward_config_4b.yaml \
  $WANDB_ARGS \
  ${OTHER_OPTION:-} \
  2>&1 | tee -a "$LOG_FILE"
