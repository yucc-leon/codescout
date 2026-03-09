#!/bin/bash
# Soni-main adapted ablation runner (migrated from acs_aditya_upstream, simplified).
#
# Why simplified:
# - Keep only ablations supported by soni/main code + existing configs.
# - Remove ablations tied to missing reward/config modules or fixed historical bugs.
#
# Kept (meaningful on soni/main):
#   A1 baseline_4b              : baseline config with multiturn reward
#   A2 vanilla_sapo             : vanilla SAPO (norm_by_std=true + eps=0.2 + sequence_mean)
#   A3 dr_grpo_pure             : Dr.GRPO pure (grpo + regular + seq_mean_token_sum_norm + no-std + eps=0.2)
#   A4 length_norm_sqrt         : grpo -> grpo_length_norm_sqrt
#   A5 sapo_on_sqrt             : gspo -> sapo under sqrt norm
#   A6 eps_clip_0.2             : eps clip 3e-4/4e-4 -> 0.2/0.2
#   A7 filtered_data            : data ablation (raw -> filtered), optional
#   A8 dr_grpo_gspo             : Dr.GRPO config combo with gspo
#   A9 dr_grpo_sapo             : Dr.GRPO config combo with sapo
#   A10 grpo_norm_by_std         : grpo_norm_by_std=true (standard GRPO std normalization)
#   A11 vanilla_grpo             : vanilla GRPO (norm_by_std=true + regular + eps=0.2 + sequence_mean)
#
# Pruned as redundant / unsupported on soni/main:
# - step3 buffer_succeed / train_tool_call: historical bug path, now fixed in generator code.
# - compliance / hierarchical / behavior_shaping / format_reward ablations:
#   related reward fns/configs do not exist in soni/main.
# - duplicated reruns (e.g. s2_r2): manual rerun can be done with RUN_NAME override.
#
# Usage:
#   bash scripts/run_ablation.sh all
#   bash scripts/run_ablation.sh a4
#   bash scripts/run_ablation.sh status
#   bash scripts/run_ablation.sh a1 --resume

set -euo pipefail
set -x

[ -f .env ] && . .env

RESUME_MODE="none"
PHASE=""
for arg in "$@"; do
  case "$arg" in
    --resume|-r) RESUME_MODE="auto" ;;
    *) if [ -z "$PHASE" ]; then PHASE="$arg"; fi ;;
  esac
done
PHASE="${PHASE:-help}"

LOG_DATE=$(date +%m%d)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

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
WANDB_API_KEY="${WANDB_API_KEY:-}"
PROJECT_NAME="${PROJECT_NAME:-soni_ablation_4b}"
CKPT_BASE="${CKPT_BASE:-ckpts}"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || true)
HALF_NUM_GPUS=$((NUM_GPUS > 0 ? NUM_GPUS / 2 : 4))
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-$HALF_NUM_GPUS}"
NUM_TRAINING_ENGINES="${NUM_TRAINING_ENGINES:-$HALF_NUM_GPUS}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"

DATA_RAW="${DATA_RAW:-data/adityasoni17__SWE-smith-py-code-search_train}"
DATA_FILTERED="${DATA_FILTERED:-data/filtered_v0204}"
[[ "$DATA_RAW" != /* ]] && DATA_RAW="$REPO_ROOT/$DATA_RAW"
[[ "$DATA_FILTERED" != /* ]] && DATA_FILTERED="$REPO_ROOT/$DATA_FILTERED"

BASE_EXP_CONFIG="${BASE_EXP_CONFIG:-configs/rewards/baseline_4b.yaml}"
NO_MULTITURN_CONFIG="${NO_MULTITURN_CONFIG:-configs/reward_config_4b.yaml}"

DECISION_FILE="logs/ablation_decisions_soni.txt"
mkdir -p logs

kill_openhands_tmux() {
  tmux list-windows -a -F '#{session_name}:#{window_index} #{window_name}' 2>/dev/null \
    | rg openhands \
    | cut -d' ' -f1 \
    | xargs -r -n1 tmux kill-window -t 2>/dev/null || true
}

cleanup() {
  uv run ray stop || true
  pkill -9 VLLM 2>/dev/null || true
  pkill -9 uv 2>/dev/null || true
  uv cache prune 2>/dev/null || true
  kill_openhands_tmux
}

is_exp_completed() {
  local run_name=$1
  local exp_dir="${CKPT_BASE}/${run_name}"
  if [ -d "$exp_dir/exported_model" ]; then
    local exported_steps
    exported_steps=$(ls -d "$exp_dir/exported_model"/global_step_* 2>/dev/null | wc -l || true)
    if [ "$exported_steps" -gt 0 ]; then
      return 0
    fi
  fi
  return 1
}

run_exp() {
  local run_name=$1
  local desc=$2
  local data_path=$3
  local exp_config=$4
  local advantage_estimator=$5
  local policy_loss_type=$6
  local loss_reduction=$7
  local eps_clip_low=$8
  local eps_clip_high=$9
  local prompt_system="${10}"
  local prompt_user="${11}"
  local extra_opts="${12:-}"

  [[ "$data_path" != /* ]] && data_path="$REPO_ROOT/$data_path"
  if [ ! -f "${data_path}/train.parquet" ]; then
    echo "Missing train data: ${data_path}/train.parquet"
    return 1
  fi
  if [ ! -f "${data_path}/validation.parquet" ]; then
    echo "Missing val data: ${data_path}/validation.parquet"
    return 1
  fi

  local ckpt_path="${CKPT_BASE}/${run_name}/"
  local log_file="logs/${LOG_DATE}_${run_name}.log"
  mkdir -p "$ckpt_path"

  local actual_resume_mode="$RESUME_MODE"
  local wandb_run_id=""
  local WANDB_ID_FILE="${ckpt_path}wandb_run_id.txt"

  if [ "$RESUME_MODE" = "auto" ]; then
    local latest_ckpt
    latest_ckpt=$(ls -d "${ckpt_path}"global_step_* 2>/dev/null | sort -t_ -k3 -rn | head -1 || true)
    if [ -n "$latest_ckpt" ] && [ -f "${latest_ckpt}/trainer_state.pt" ]; then
      actual_resume_mode="latest"
      if [ -f "$WANDB_ID_FILE" ]; then
        wandb_run_id=$(cat "$WANDB_ID_FILE")
        echo "Resuming wandb run: $wandb_run_id"
      fi
    else
      actual_resume_mode="none"
    fi
  elif [ "$RESUME_MODE" = "latest" ] && [ -f "$WANDB_ID_FILE" ]; then
    wandb_run_id=$(cat "$WANDB_ID_FILE")
    echo "Resuming wandb run: $wandb_run_id"
  fi

  local wandb_args=""
  [ -n "$wandb_run_id" ] && wandb_args="+trainer.wandb_run_id=${wandb_run_id}"

  {
    echo "=========================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Experiment: $run_name"
    echo "Description: $desc"
    echo "Data: $data_path"
    echo "Config: $exp_config"
    echo "Adv/Policy/Loss: $advantage_estimator / $policy_loss_type / $loss_reduction"
    echo "Eps clip: $eps_clip_low / $eps_clip_high"
    echo "Resume: $actual_resume_mode"
    echo "=========================================="
  } | tee -a "$log_file"

  uv run --isolated -m src.train \
    +run_async_trainer=true \
    "data.train_data=['${data_path}/train.parquet']" \
    "data.val_data=['${data_path}/validation.parquet']" \
    trainer.algorithm.advantage_estimator="${advantage_estimator}" \
    trainer.algorithm.policy_loss_type="${policy_loss_type}" \
    trainer.algorithm.loss_reduction="${loss_reduction}" \
    trainer.algorithm.eps_clip_low="${eps_clip_low}" \
    trainer.algorithm.eps_clip_high="${eps_clip_high}" \
    trainer.algorithm.grpo_norm_by_std=false \
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
    +generator.traj_dir="${ckpt_path}trajectories/" \
    +generator.exp_config="${exp_config}" \
    +generator.prompts.system_prompt="${prompt_system}" \
    +generator.prompts.user_prompt="${prompt_user}" \
    +generator.engine_init_kwargs.enable_auto_tool_choice=true \
    +generator.engine_init_kwargs.tool_call_parser=hermes \
    +generator.engine_init_kwargs.max_model_len=40960 \
    +generator.engine_init_kwargs.disable_cascade_attn=true \
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
    +trainer.save_session_sample_every_step=true \
    +trainer.session_sample_interval=10 \
    +trainer.session_sample_num_instances=3 \
    trainer.export_path="${ckpt_path}exported_model/" \
    trainer.hf_save_interval=25 \
    trainer.ckpt_interval=100 \
    trainer.use_sample_packing=false \
    trainer.max_prompt_length=40960 \
    generator.sampling_params.max_generate_length="${MAX_LENGTH}" \
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
    generator.n_samples_per_prompt="${N_ROLLOUTS}" \
    generator.gpu_memory_utilization=0.8 \
    generator.enforce_eager=false \
    +trainer.step_wise_training=false \
    trainer.logger=wandb \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.run_name="${run_name}" \
    trainer.resume_mode="${actual_resume_mode}" \
    trainer.ckpt_path="${ckpt_path}" \
    trainer.max_ckpts_to_keep=5 \
    ${wandb_args} \
    ${extra_opts} 2>&1 | tee -a "$log_file"
}

run_a1_baseline() {
  run_exp \
    "a1_baseline_4b" \
    "Baseline on soni/main (multilevel + multiturn)" \
    "$DATA_RAW" \
    "$BASE_EXP_CONFIG" \
    "grpo" \
    "gspo" \
    "sequence_mean" \
    "3e-4" \
    "4e-4" \
    "templates/system_prompt_custom_finish.j2" \
    "templates/file_module_custom_finish.j2"
} # running

run_a2_sapo() {
  run_exp \
    "a2_sapo" \
    "Vanilla SAPO: norm_by_std=true + sapo + eps=0.2 + sequence_mean" \
    "$DATA_RAW" \
    "$BASE_EXP_CONFIG" \
    "grpo" \
    "sapo" \
    "sequence_mean" \
    "0.2" \
    "0.2" \
    "templates/system_prompt_custom_finish.j2" \
    "templates/file_module_custom_finish.j2" \
    "trainer.algorithm.grpo_norm_by_std=true"
}

run_a3_dr_grpo_pure() {
  run_exp \
    "a3_dr_grpo_pure" \
    "Dr.GRPO pure: grpo + regular + seq_mean_token_sum_norm + no-std + eps=0.2" \
    "$DATA_RAW" \
    "$BASE_EXP_CONFIG" \
    "grpo" \
    "regular" \
    "seq_mean_token_sum_norm" \
    "0.2" \
    "0.2" \
    "templates/system_prompt_custom_finish.j2" \
    "templates/file_module_custom_finish.j2"
}

run_a4_length_norm_sqrt() {
  run_exp \
    "a4_length_norm_sqrt" \
    "Advantage ablation: grpo_length_norm_sqrt" \
    "$DATA_RAW" \
    "$BASE_EXP_CONFIG" \
    "grpo_length_norm_sqrt" \
    "gspo" \
    "seq_mean_token_sum_norm" \
    "3e-4" \
    "4e-4" \
    "templates/system_prompt_custom_finish.j2" \
    "templates/file_module_custom_finish.j2"
}

run_a5_sapo_on_sqrt() {
  run_exp \
    "a5_sapo_on_sqrt" \
    "Policy ablation: sapo + sqrt norm" \
    "$DATA_RAW" \
    "$BASE_EXP_CONFIG" \
    "grpo_length_norm_sqrt" \
    "sapo" \
    "seq_mean_token_sum_norm" \
    "3e-4" \
    "4e-4" \
    "templates/system_prompt_custom_finish.j2" \
    "templates/file_module_custom_finish.j2"
}

run_a6_eps_clip_02() {
  run_exp \
    "a6_eps_clip_02" \
    "Eps clip ablation: 0.2 / 0.2" \
    "$DATA_RAW" \
    "$BASE_EXP_CONFIG" \
    "grpo_length_norm_sqrt" \
    "gspo" \
    "seq_mean_token_sum_norm" \
    "0.2" \
    "0.2" \
    "templates/system_prompt_custom_finish.j2" \
    "templates/file_module_custom_finish.j2"
}

run_a7_filtered_data() {
  if [ ! -f "${DATA_FILTERED}/train.parquet" ]; then
    echo "Skip A7: filtered data not found at ${DATA_FILTERED}/train.parquet"
    return 0
  fi
  run_exp \
    "a7_filtered_data" \
    "Data ablation: filtered dataset" \
    "$DATA_FILTERED" \
    "$BASE_EXP_CONFIG" \
    "grpo_length_norm_sqrt" \
    "gspo" \
    "seq_mean_token_sum_norm" \
    "3e-4" \
    "4e-4" \
    "templates/system_prompt_custom_finish.j2" \
    "templates/file_module_custom_finish.j2"
}

run_a8_dr_grpo_gspo() {
  run_exp \
    "a8_dr_grpo_gspo" \
    "Dr.GRPO config combo: grpo + no-std + seq_mean_token_sum_norm + gspo" \
    "$DATA_RAW" \
    "$BASE_EXP_CONFIG" \
    "grpo" \
    "gspo" \
    "seq_mean_token_sum_norm" \
    "3e-4" \
    "4e-4" \
    "templates/system_prompt_custom_finish.j2" \
    "templates/file_module_custom_finish.j2"
}

run_a9_dr_grpo_sapo() {
  run_exp \
    "a9_dr_grpo_sapo" \
    "Dr.GRPO config combo: grpo + no-std + seq_mean_token_sum_norm + sapo" \
    "$DATA_RAW" \
    "$BASE_EXP_CONFIG" \
    "grpo" \
    "sapo" \
    "seq_mean_token_sum_norm" \
    "3e-4" \
    "4e-4" \
    "templates/system_prompt_custom_finish.j2" \
    "templates/file_module_custom_finish.j2"
}

run_a10_grpo_norm_by_std() {
  run_exp \
    "a10_grpo_norm_by_std" \
    "Advantage ablation: grpo_norm_by_std=true (standard GRPO std normalization)" \
    "$DATA_RAW" \
    "$BASE_EXP_CONFIG" \
    "grpo" \
    "gspo" \
    "sequence_mean" \
    "3e-4" \
    "4e-4" \
    "templates/system_prompt_custom_finish.j2" \
    "templates/file_module_custom_finish.j2" \
    "trainer.algorithm.grpo_norm_by_std=true"
}

run_a11_vanilla_grpo() {
  run_exp \
    "a11_vanilla_grpo" \
    "Vanilla GRPO: norm_by_std=true + regular PPO clip + eps=0.2 + sequence_mean" \
    "$DATA_RAW" \
    "$BASE_EXP_CONFIG" \
    "grpo" \
    "regular" \
    "sequence_mean" \
    "0.2" \
    "0.2" \
    "templates/system_prompt_custom_finish.j2" \
    "templates/file_module_custom_finish.j2" \
    "trainer.algorithm.grpo_norm_by_std=true"
}

show_status() {
  echo "=== soni/main ablation status ==="
  local exps=(
    a1_baseline_4b
    a2_sapo
    a3_dr_grpo_pure
    a4_length_norm_sqrt
    a5_sapo_on_sqrt
    a6_eps_clip_02
    a7_filtered_data
    a8_dr_grpo_gspo
    a9_dr_grpo_sapo
    a10_grpo_norm_by_std
    a11_vanilla_grpo
  )
  for exp in "${exps[@]}"; do
    if is_exp_completed "$exp"; then
      echo "  ✅ $exp"
    else
      echo "  ⏳ $exp"
    fi
  done
  echo
  echo "Raw data: $DATA_RAW"
  echo "Filtered data: $DATA_FILTERED"
}

run_all() {
  cleanup
  run_a1_baseline
  cleanup
  run_a2_sapo
  cleanup
  run_a3_dr_grpo_pure
  cleanup
  run_a4_length_norm_sqrt
  cleanup
  run_a5_sapo_on_sqrt
  cleanup
  run_a6_eps_clip_02
  cleanup
  run_a7_filtered_data
  cleanup
  run_a8_dr_grpo_gspo
  cleanup
  run_a9_dr_grpo_sapo
  cleanup
  run_a10_grpo_norm_by_std
  cleanup
  run_a11_vanilla_grpo
}

case "$PHASE" in
  all) run_all ;;
  a1|baseline) cleanup; run_a1_baseline ;;
  a2|_sapo) cleanup; run_a2_sapo ;;
  a3|dr_grpo_pure) cleanup; run_a3_dr_grpo_pure ;;
  a4|sqrt) cleanup; run_a4_length_norm_sqrt ;;
  a5|sapo) cleanup; run_a5_sapo_on_sqrt ;;
  a6|epsclip) cleanup; run_a6_eps_clip_02 ;;
  a7|filtered) cleanup; run_a7_filtered_data ;;
  a8|drgrpo) cleanup; run_a8_dr_grpo_gspo ;;
  a9|drgrpo_sapo) cleanup; run_a9_dr_grpo_sapo ;;
  a10|norm_by_std) cleanup; run_a10_grpo_norm_by_std ;;
  a11|vanilla_grpo) cleanup; run_a11_vanilla_grpo ;;
  status|--status|-s) show_status ;;
  help|--help|-h)
    cat <<'EOF'
Usage: bash scripts/run_ablation.sh <command> [--resume]

Commands:
  all          run all kept ablations (A1-A11)
  a1           baseline (multilevel + multiturn)
  a2           vanilla SAPO (grpo + sapo + std norm + eps 0.2)
  a3           Dr.GRPO pure (grpo + regular + seq_mean_token_sum_norm + no-std + eps 0.2)
  a4           length norm sqrt
  a5           sapo on sqrt norm
  a6           eps clip 0.2/0.2
  a7           filtered data ablation (if filtered data exists)
  a8           Dr.GRPO equivalent (gspo)
  a9           Dr.GRPO equivalent (sapo)
  a10          grpo_norm_by_std=true (std normalization)
  a11          vanilla GRPO (full original paper config)
  status       check completion status

Options:
  --resume, -r auto-resume from latest checkpoint if present
EOF
    ;;
  *)
    echo "Unknown command: $PHASE"
    exit 1
    ;;
esac
