#!/bin/bash
# Baseline 4B training script.
# Uses only parameters and configs valid on soni/main (no acs-only session_sample, etc.).

# Optional: load env (WANDB_API_KEY, MODEL, etc.). Skip if not present.
[ -f .env ] && . .env

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/cache/triton}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
export TMPDIR="${TMPDIR:-/tmp}"
export TESTBED_ROOT="${TESTBED_ROOT:-/tmp/testbed}"
# WANDB: set WANDB_API_KEY in .env or environment before running
export WANDB_API_KEY="${WANDB_API_KEY:-}"

uv run ray stop || true
pkill -9 VLLM 2>/dev/null || true
pkill -9 uv 2>/dev/null || true
uv cache prune 2>/dev/null || true
tmux list-windows -a -F '#{session_name}:#{window_index} #{window_name}' 2>/dev/null \
    | grep openhands \
    | cut -d' ' -f1 \
    | xargs -r -n1 tmux kill-window -t 2>/dev/null || true

# Model path: override via MODEL env or set default
MODEL="${BASE_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
N_ROLLOUTS="${N_ROLLOUTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH=8192
RUN_NAME="${RUN_NAME:-Baseline-4B-2507-sonibugfix-withrg}"
LOG_DATE=$(date +%m%d)
LOG_FILE="logs/${LOG_DATE}_${RUN_NAME}.log"
set -x

DATA_PATH="${DATA_PATH:-data/adityasoni17__SWE-smith-py-code-search_train}"
[[ "$DATA_PATH" != /* ]] && DATA_PATH="$(cd "$(dirname "$0")/.." && pwd)/${DATA_PATH}"
if [ ! -f "${DATA_PATH}/train.parquet" ]; then
  echo "Error: train data not found: ${DATA_PATH}/train.parquet"
  echo "Set DATA_PATH or place train.parquet (and validation.parquet) in data/adityasoni17__SWE-smith-py-code-search_train/."
  exit 1
fi

CKPT_PATH="${CKPT_PATH:-ckpts/${RUN_NAME}}"
LCAL_PATH="${LCAL_PATH:-$CKPT_PATH}"
mkdir -p "$CKPT_PATH"
mkdir -p "$(cd "$(dirname "$0")/.." && pwd)/logs"

# 续训依赖：SkyRL 的 resume_mode=latest 只认 ckpt_path 下的 latest_ckpt_global_step.txt。
# 若目录里已有 global_step_*/trainer_state.pt 但该文件缺失（例如从别处拷来权重），这里补写以便 trainer 能续训。
LATEST_CKPT_FILE="${LCAL_PATH}/latest_ckpt_global_step.txt"
if [ ! -f "$LATEST_CKPT_FILE" ]; then
  latest_ckpt=$(ls -d "${LCAL_PATH}"/global_step_* 2>/dev/null | sort -t_ -k3 -rn | head -1)
  if [ -n "$latest_ckpt" ] && [ -f "${latest_ckpt}/trainer_state.pt" ]; then
    step_num=$(basename "$latest_ckpt" | sed 's/global_step_//')
    echo "$step_num" > "$LATEST_CKPT_FILE"
    echo "Created $LATEST_CKPT_FILE -> $step_num (so trainer can resume from existing checkpoint)"
  fi
fi

# 与 run_baseline_14b 一致：根据 ckpt 目录决定是否续训，并加载 wandb run id 以接续曲线
RESUME_MODE="${RESUME_MODE:-latest}"
actual_resume_mode="$RESUME_MODE"
wandb_run_id=""
WANDB_ID_FILE="${LCAL_PATH}/wandb_run_id.txt"

if [ "$RESUME_MODE" = "latest" ]; then
  if [ -f "$LATEST_CKPT_FILE" ]; then
    step_num=$(cat "$LATEST_CKPT_FILE" 2>/dev/null | tr -d '\n')
    ckpt_dir="${LCAL_PATH}/global_step_${step_num}"
    if [ -d "$ckpt_dir" ] && [ -f "${ckpt_dir}/trainer_state.pt" ]; then
      echo "Found checkpoint: global_step $step_num, will resume (ckpt_path=$LCAL_PATH)"
      if [ -f "$WANDB_ID_FILE" ]; then
        wandb_run_id=$(cat "$WANDB_ID_FILE")
        echo "Resuming wandb run: $wandb_run_id"
      else
        echo "No wandb_run_id.txt found, will create new wandb run"
      fi
    else
      echo "WARNING: $LATEST_CKPT_FILE points to step $step_num but ${ckpt_dir} or trainer_state.pt missing; trainer may start from scratch"
    fi
  else
    echo "No latest_ckpt_global_step.txt and no valid global_step_* in $LCAL_PATH; starting from scratch"
    actual_resume_mode="none"
  fi
fi

wandb_args=""
[ -n "$wandb_run_id" ] && wandb_args="+trainer.wandb_run_id=${wandb_run_id}"

# 写入日志：续训/从头训一目了然
if [ "$actual_resume_mode" = "latest" ]; then
  _step=$(cat "$LATEST_CKPT_FILE" 2>/dev/null | tr -d '\n')
  if [ -n "$_step" ]; then
    echo "[$(date -Iseconds)] ===== CHECKPOINT RESUME: yes, from global_step ${_step} (ckpt_path=$LCAL_PATH) =====" >> "$LOG_FILE"
  else
    echo "[$(date -Iseconds)] ===== CHECKPOINT RESUME: latest (ckpt_path=$LCAL_PATH, trainer will resolve step) =====" >> "$LOG_FILE"
  fi
else
  echo "[$(date -Iseconds)] ===== CHECKPOINT RESUME: no (starting from scratch, ckpt_path=$LCAL_PATH, resume_mode=$actual_resume_mode) =====" >> "$LOG_FILE"
fi

HALF_NUM_GPUS=$((NUM_GPUS > 0 ? NUM_GPUS / 2 : 4))
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-$HALF_NUM_GPUS}"
NUM_TRAINING_ENGINES="${NUM_TRAINING_ENGINES:-$HALF_NUM_GPUS}"
STEP_WISE="${STEP_WISE:-false}"

export VLLM_FLASH_ATTN_VERSION=2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export RAY_worker_register_timeout_seconds=600

# Align with run_async_training_1.7b.sh: no exp_config.
# train.py defaults: reward=[multilevel_localization_f1_reward], tools=[terminal].
# Generator code always uses terminal + localization_finish; prompts set via CLI below.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

uv run --isolated -m src.train \
  +run_async_trainer=true \
  data.train_data="['$DATA_PATH/train.parquet']" \
  data.val_data="['$DATA_PATH/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.policy.model.path="${MODEL}" \
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
  trainer.policy.sequence_parallel_size=1 \
  generator.num_inference_engines=${NUM_INFERENCE_ENGINES} \
  generator.inference_engine_tensor_parallel_size=1 \
  +generator.traj_dir="${CKPT_PATH}/trajectories/" \
  +generator.engine_init_kwargs.enable_auto_tool_choice=true \
  +generator.engine_init_kwargs.tool_call_parser=hermes \
  +generator.engine_init_kwargs.max_model_len=40960 \
  +generator.prompts.system_prompt="templates/system_prompt_custom_finish.j2" \
  +generator.prompts.user_prompt="templates/file_module_custom_finish.j2" \
  +generator.engine_init_kwargs.disable_cascade_attn=true \
  generator.eval_n_samples_per_prompt=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=true \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${BATCH_SIZE} \
  trainer.policy_mini_batch_size=${BATCH_SIZE} \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
  trainer.dump_data_batch=true \
  trainer.export_path="${CKPT_PATH}/exported_model/" \
  trainer.hf_save_interval=25 \
  trainer.ckpt_interval=50 \
  trainer.use_sample_packing=false \
  trainer.max_prompt_length=40960 \
  trainer.algorithm.policy_loss_type="gspo" \
  trainer.algorithm.eps_clip_low=3e-4 \
  trainer.algorithm.eps_clip_high=4e-4 \
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
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.n_samples_per_prompt=${N_ROLLOUTS} \
  generator.gpu_memory_utilization=0.8 \
  generator.enforce_eager=false \
  +trainer.step_wise_training=${STEP_WISE} \
  trainer.logger="wandb" \
  trainer.project_name="soni_baseline_4b" \
  trainer.run_name="${RUN_NAME}" \
  trainer.resume_mode="${actual_resume_mode}" \
  trainer.ckpt_path="${LCAL_PATH}" \
  trainer.max_ckpts_to_keep=5 \
  +generator.exp_config=configs/rewards/baseline_4b.yaml \
  ${wandb_args} \
  ${OTHER_OPTION:+"$OTHER_OPTION"} 2>&1 | tee -a "$LOG_FILE"
