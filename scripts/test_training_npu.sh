#!/bin/bash
# Minimal NPU training test on CANN 8.3
# Usage: bash codescout/scripts/test_training_npu.sh
set -euo pipefail

# === CANN environment ===
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash 2>/dev/null || true
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
export HCCL_CONNECT_TIMEOUT=360
export HCCL_EXEC_TIMEOUT=360
export HCCL_IF_BASE_PORT=64033
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_NZ=0

# === General ===
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_worker_register_timeout_seconds=600
export PYTHONUNBUFFERED=1
export SKYRL_PYTHONPATH_EXPORT=1
export SKYRL_LD_LIBRARY_PATH_EXPORT=1
export WANDB_MODE=offline

# === Cache dirs ===
export TORCHINDUCTOR_CACHE_DIR=/tmp/cache/torchinductor
export TRITON_CACHE_DIR=/tmp/cache/triton
export RAY_TMPDIR=/tmp/ray
export TMPDIR=/tmp
export TESTBED_ROOT=/tmp/testbed
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$TESTBED_ROOT"

# === Conda / Python ===
CONDA_ENV=${CONDA_ENV:-$HOME/miniforge3/envs/codescout-cann83}
PYTHON="$CONDA_ENV/bin/python"
CODESCOUT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKYRL_ROOT="$(dirname "$CODESCOUT_ROOT")/SkyRL"
export PYTHONPATH="$CODESCOUT_ROOT:$SKYRL_ROOT:${PYTHONPATH:-}"

[ -f "$CODESCOUT_ROOT/.env" ] && source "$CODESCOUT_ROOT/.env"

# === Params ===
MODEL=${MODEL:-/path/to/Qwen3-4B-Instruct-2507}
DATA=$CODESCOUT_ROOT/data/swe_smith
CKPT=/tmp/codescout_test_ckpts
mkdir -p "$CKPT" logs

# === Cleanup ===
cleanup() {
    echo "🧹 Cleaning up..."
    ray stop 2>/dev/null || true
    sleep 3
}
trap cleanup EXIT

echo "===================="
echo "🚀 NPU Training Test"
echo "📦 Model: $MODEL"
echo "===================="

$PYTHON -m src.train_npu \
  +run_async_trainer=true \
  "data.train_data=['$DATA/train.parquet']" \
  "data.val_data=['$DATA/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.policy.model.path=$MODEL \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.policy.fsdp_config.reshard_after_forward=true \
  trainer.policy.fsdp_config.fsdp_size=-1 \
  trainer.placement.policy_num_gpus_per_node=2 \
  trainer.placement.ref_num_gpus_per_node=2 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.policy.sequence_parallel_size=1 \
  trainer.fully_async.num_parallel_generation_workers=2 \
  generator.num_inference_engines=2 \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.enable_http_endpoint=True \
  generator.http_endpoint_host='0.0.0.0' \
  generator.http_endpoint_port=8080 \
  generator.weight_sync_backend=hccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.n_samples_per_prompt=1 \
  generator.gpu_memory_utilization=0.7 \
  generator.enforce_eager=true \
  generator.max_input_length=4096 \
  generator.max_num_batched_tokens=8192 \
  generator.max_turns=3 \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1.0 \
  +generator.engine_init_kwargs.enable_auto_tool_choice=true \
  +generator.engine_init_kwargs.tool_call_parser=hermes \
  +generator.engine_init_kwargs.max_model_len=4096 \
  +generator.engine_init_kwargs.disable_cascade_attn=true \
  +generator.prompts.system_prompt=templates/system_prompt_custom_finish.j2 \
  +generator.prompts.user_prompt=templates/file_module_custom_finish.j2 \
  +generator.traj_dir=$CKPT/trajectories/ \
  +generator.reward=configs/reward_config_4b.yaml \
  trainer.flash_attn=false \
  trainer.epochs=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=4096 \
  trainer.hf_save_interval=-1 \
  trainer.ckpt_interval=999 \
  trainer.use_sample_packing=false \
  trainer.algorithm.policy_loss_type=gspo \
  trainer.algorithm.eps_clip_low=0.0003 \
  trainer.algorithm.eps_clip_high=0.0004 \
  trainer.algorithm.loss_reduction=sequence_mean \
  trainer.algorithm.use_kl_loss=False \
  trainer.algorithm.use_kl_in_reward=False \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.logger=console \
  trainer.project_name=codescout_npu_test \
  trainer.run_name=cann83-4b-test \
  trainer.resume_mode=none \
  trainer.ckpt_path=$CKPT \
  trainer.dump_data_batch=true \
  2>&1 | tee logs/test_training_npu.log
