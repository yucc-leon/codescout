#!/bin/bash
# Ablation V2: 贪心爬坡消融实验
#
# 设计原则：逐项确定最佳参数，后续 step 复用前序 step 的胜者
#
# 实验矩阵（无重复）：
#   Step 1: prompt 消融
#     - s1_prompt_old: old prompt + beta=0.5 + raw (预期失败)
#     - s1_prompt_new: tool_only + beta=0.5 + raw (baseline)
#   Step 2: beta 消融（复用 s1_prompt_new 作为 beta=0.5）
#     - s2_beta10: tool_only + beta=1.0 + raw
#   Step 3: 数据消融（使用 step2 选定的 beta）
#     - s3_filtered: tool_only + (beta) + filtered
#   Step 4: reward 函数消融（使用 step2,3 选定的配置）
#     - s4_hierarchical: hierarchical + (beta) + (data)
#
# 用法:
#   bash scripts/run_ablation.sh step1              # 从头开始
#   bash scripts/run_ablation.sh step1 --resume     # 续训（自动恢复 checkpoint 和 wandb）
#   bash scripts/run_ablation.sh s1_new --resume    # 单独续训某个实验
#   bash scripts/run_ablation.sh --status
#   bash scripts/run_ablation.sh --decide step1

set -x

# 解析参数
RESUME_MODE="none"
PHASE=""
for arg in "$@"; do
    case "$arg" in
        --resume|-r)
            RESUME_MODE="auto"
            ;;
        *)
            if [ -z "$PHASE" ]; then
                PHASE="$arg"
            fi
            ;;
    esac
done
PHASE="${PHASE:-help}"
LOG_DATE=$(date +%m%d)

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/cache/triton}"


# ========== 环境变量 ==========
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
export TMPDIR="${TMPDIR:-/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export TESTBED_ROOT="${TESTBED_ROOT:-/tmp/testbed}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$TESTBED_ROOT" logs

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH

rm -rf ~/.cache/flashinfer
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_FLASH_ATTN_VERSION=2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0

# ========== 4B 模型配置 ==========
export BASE_MODEL="Qwen3-4B-Instruct-2507"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to your local model directory}"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY env var before running}"
export PROJECT_NAME="ablation_v2_4b"
CKPT_BASE="${CKPT_BASE:-/tmp/ckpts}"

# 数据路径 (v0204: 从预过滤 6759 样本池采样，同样本数、同 seed)
DATA_RAW="${DATA_RAW:-data/raw_v0204}"
DATA_FILTERED="${DATA_FILTERED:-data/filtered_v0204}"

TRAIN_BATCH_SIZE=16
MICRO_BATCH_SIZE=4
MICRO_TRAIN_BATCH_SIZE=1
NUM_TRAINING_ENGINES=6
NUM_INFERENCE_ENGINES=2
NUM_PARALLEL_GENERATION_WORKERS=16

# ========== 决策记录 ==========
DECISION_FILE="logs/ablation_decisions.txt"

get_decision() {
    local step=$1
    if [ -f "$DECISION_FILE" ]; then
        grep "^${step}=" "$DECISION_FILE" | cut -d'=' -f2
    fi
}

set_decision() {
    local step=$1
    local value=$2
    mkdir -p logs
    if [ -f "$DECISION_FILE" ]; then
        grep -v "^${step}=" "$DECISION_FILE" > "${DECISION_FILE}.tmp" || true
        mv "${DECISION_FILE}.tmp" "$DECISION_FILE"
    fi
    echo "${step}=${value}" >> "$DECISION_FILE"
    echo "✅ 已记录决策: ${step}=${value}"
}

# ========== 获取当前最佳配置 ==========
get_best_config() {
    local beta=$(get_decision "step2_beta")
    local data=$(get_decision "step3_data")
    
    # 默认值
    beta="${beta:-0.5}"
    data="${data:-raw}"
    
    echo "$beta $data"
}

# ========== 检查实验状态 ==========
is_exp_completed() {
    local run_name=$1
    local exp_dir="${CKPT_BASE}/${run_name}-${BASE_MODEL}"
    
    # 检查 exported_model 下是否有 global_step_* 目录（真正的导出模型）
    if [ -d "$exp_dir/exported_model" ]; then
        local exported_steps=$(ls -d "$exp_dir/exported_model"/global_step_* 2>/dev/null | wc -l)
        if [ "$exported_steps" -gt 0 ]; then
            return 0
        fi
    fi
    
    # 检查是否有足够的 checkpoint
    local max_step=$(ls -d "$exp_dir"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -rn | head -1)
    if [ -n "$max_step" ] && [ "$max_step" -ge 100 ]; then
        return 0
    fi
    
    return 1
}

# ========== 运行实验 ==========
run_exp() {
    local run_name=$1
    local config_path=$2
    local data_path=$3
    local description=$4
    local skip_if_done=${5:-true}
    
    local CKPT_PATH="${CKPT_BASE}/${run_name}-${BASE_MODEL}/"
    local LOG_FILE="logs/${LOG_DATE}_${run_name}.log"
    
    if [ "$skip_if_done" = "true" ] && is_exp_completed "$run_name"; then
        echo "⏭️  Skipping $run_name: already completed"
        return 0
    fi
    
    if [ ! -f "${data_path}/train.parquet" ]; then
        echo "❌ 数据不存在: ${data_path}/train.parquet"
        return 1
    fi
    
    # 确定实际的 resume_mode
    local actual_resume_mode="$RESUME_MODE"
    local wandb_run_id=""
    local WANDB_ID_FILE="${CKPT_PATH}wandb_run_id.txt"
    
    if [ "$RESUME_MODE" = "auto" ]; then
        # 检查是否有可恢复的 checkpoint
        local latest_ckpt=$(ls -d "${CKPT_PATH}"global_step_* 2>/dev/null | sort -t_ -k3 -rn | head -1)
        if [ -n "$latest_ckpt" ] && [ -f "${latest_ckpt}/trainer_state.pt" ]; then
            local step_num=$(basename "$latest_ckpt" | sed 's/global_step_//')
            echo "🔄 Found checkpoint: step $step_num, resuming..."
            # 读取 wandb run id
            if [ -f "$WANDB_ID_FILE" ]; then
                wandb_run_id=$(cat "$WANDB_ID_FILE")
                echo "🔗 Resuming wandb run: $wandb_run_id"
            else
                echo "⚠️  No wandb_run_id.txt found, will create new wandb run"
            fi
        else
            echo "⚠️  No valid checkpoint found, starting fresh"
            actual_resume_mode="none"
        fi
    fi
    
    {
        echo "=========================================="
        echo "📅 $(date '+%Y-%m-%d %H:%M:%S')"
        echo "🚀 Experiment: $run_name"
        echo "📝 $description"
        echo "⚙️  Config: $config_path"
        echo "📂 Data: $data_path"
        echo "🔄 Resume mode: $actual_resume_mode"
        [ -n "$wandb_run_id" ] && echo "🔗 Wandb run: $wandb_run_id"
        echo "=========================================="
    } 2>&1 | tee -a "$LOG_FILE"
    
    mkdir -p "$CKPT_PATH"
    
    # 构建 wandb 参数
    local wandb_args=""
    if [ -n "$wandb_run_id" ]; then
        wandb_args="+trainer.wandb_run_id=${wandb_run_id}"
    fi

    uv run --isolated -m src.train \
        +run_async_trainer=true \
        "data.train_data=['${data_path}/train.parquet']" \
        "data.val_data=['${data_path}/validation.parquet']" \
        trainer.algorithm.advantage_estimator=grpo \
        trainer.algorithm.policy_loss_type=gspo \
        trainer.algorithm.eps_clip_low=3e-4 \
        trainer.algorithm.eps_clip_high=4e-4 \
        trainer.algorithm.loss_reduction=sequence_mean \
        trainer.algorithm.grpo_norm_by_std=false \
        trainer.policy.model.path=${BASE_MODEL_PATH} \
        trainer.placement.colocate_all=false \
        trainer.placement.colocate_policy_ref=true \
        +generator.engine_init_kwargs.disable_cascade_attn=true \
        trainer.strategy=fsdp2 \
        trainer.policy.fsdp_config.cpu_offload=true \
        trainer.policy.fsdp_config.reshard_after_forward=true \
        trainer.policy.fsdp_config.fsdp_size=-1 \
        trainer.fully_async.num_parallel_generation_workers=${NUM_PARALLEL_GENERATION_WORKERS} \
        trainer.placement.policy_num_gpus_per_node=${NUM_TRAINING_ENGINES} \
        trainer.placement.ref_num_gpus_per_node=${NUM_TRAINING_ENGINES} \
        trainer.placement.policy_num_nodes=1 \
        trainer.placement.ref_num_nodes=1 \
        trainer.policy.sequence_parallel_size=1 \
        generator.num_inference_engines=${NUM_INFERENCE_ENGINES} \
        generator.inference_engine_tensor_parallel_size=1 \
        +generator.traj_dir=${CKPT_PATH}trajectories/ \
        +generator.exp_config=${config_path} \
        +generator.engine_init_kwargs.enable_auto_tool_choice=true \
        +generator.engine_init_kwargs.tool_call_parser=hermes \
        +generator.engine_init_kwargs.rope_scaling.rope_type=yarn \
        +generator.engine_init_kwargs.rope_scaling.factor=2.0 \
        +generator.engine_init_kwargs.rope_scaling.original_max_position_embeddings=32768 \
        +generator.engine_init_kwargs.max_model_len=64000 \
        trainer.use_sample_packing=false \
        trainer.epochs=1 \
        trainer.eval_batch_size=16 \
        trainer.eval_before_train=true \
        trainer.eval_interval=-1 \
        trainer.update_epochs_per_batch=1 \
        trainer.train_batch_size=${TRAIN_BATCH_SIZE} \
        trainer.policy_mini_batch_size=${TRAIN_BATCH_SIZE} \
        trainer.micro_forward_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
        trainer.micro_train_batch_size_per_gpu=${MICRO_TRAIN_BATCH_SIZE} \
        trainer.dump_data_batch=true \
        trainer.export_path=${CKPT_PATH}exported_model/ \
        trainer.hf_save_interval=25 \
        trainer.ckpt_interval=25 \
        trainer.max_prompt_length=16384 \
        generator.sampling_params.max_generate_length=4096 \
        generator.sampling_params.temperature=1.0 \
        generator.max_input_length=16384 \
        generator.max_num_batched_tokens=65536 \
        generator.max_turns=10 \
        trainer.policy.optimizer_config.lr=1.0e-6 \
        trainer.algorithm.use_kl_loss=False \
        generator.backend=vllm \
        generator.run_engines_locally=True \
        generator.enable_http_endpoint=True \
        generator.http_endpoint_host=0.0.0.0 \
        generator.http_endpoint_port=8080 \
        generator.weight_sync_backend=nccl \
        generator.async_engine=true \
        generator.batched=false \
        generator.n_samples_per_prompt=8 \
        generator.gpu_memory_utilization=0.80 \
        generator.enforce_eager=false \
        trainer.step_wise_training=true \
        trainer.logger=wandb \
        trainer.project_name=${PROJECT_NAME} \
        trainer.run_name=${run_name}-${BASE_MODEL} \
        trainer.resume_mode=${actual_resume_mode} \
        trainer.ckpt_path=${CKPT_PATH} \
        trainer.max_ckpts_to_keep=3 \
        $wandb_args \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "✅ Completed: $run_name" | tee -a "$LOG_FILE"
}

cleanup() {
    echo "🧹 Cleaning up..."
    ray stop || true
    pkill -9 VLLM || true
    uv cache prune
    sleep 5
}

# ========== Step 1: Prompt 消融 ==========
step1() {
    echo "=========================================="
    echo "Step 1: Prompt 版本消融"
    echo ""
    echo "实验:"
    echo "  s1_prompt_old: 旧 prompt (预期 reward=0)"
    echo "  s1_prompt_new: 新 prompt (baseline, beta=0.5)"
    echo ""
    echo "早停规则: 连续 30 步 localization_reward=0 可手动停止"
    echo "=========================================="
    
    echo "实验 1a: 旧 prompt"
    run_exp "s1_prompt_old" \
        "configs/skyrl-experiments/ablation/prompt_backticks_baseline.yaml" \
        "$DATA_RAW" \
        "Step1: Old prompt (expected to fail)"
    
    cleanup
    
    echo "实验 1b: 新 prompt (baseline)"
    run_exp "s1_prompt_new" \
        "configs/skyrl-experiments/ablation/multilevel_f05.yaml" \
        "$DATA_RAW" \
        "Step1: New prompt + beta=0.5 (baseline)"
    
    echo ""
    echo "=========================================="
    echo "Step 1 完成！"
    echo ""
    echo "预期结果:"
    echo "  s1_prompt_old: localization_reward ≈ 0"
    echo "  s1_prompt_new: localization_reward > 0"
    echo ""
    echo "下一步: bash scripts/run_ablation.sh --decide step1"
    echo "=========================================="
}

# ========== Step 2: Beta 消融 ==========
step2() {
    local prompt_choice=$(get_decision "step1_prompt")
    
    if [ -z "$prompt_choice" ]; then
        echo "❌ 请先完成 Step 1 并做决策"
        return 1
    fi
    
    if [ "$prompt_choice" != "tool_only" ]; then
        echo "⚠️  Step 1 选择了 old prompt，无法继续消融"
        return 1
    fi
    
    echo "=========================================="
    echo "Step 2: Beta 值消融"
    echo ""
    echo "复用 s1_prompt_new 作为 beta=0.5 baseline"
    echo "只需跑 s2_beta10 (beta=1.0)"
    echo ""
    echo "对比: s1_prompt_new (beta=0.5) vs s2_beta10 (beta=1.0)"
    echo "=========================================="
    
    echo "实验 2: beta=1.0"
    run_exp "s2_beta10" \
        "configs/skyrl-experiments/ablation/multilevel_f1.yaml" \
        "$DATA_RAW" \
        "Step2: beta=1.0 (compare with s1_prompt_new)"
    
    echo ""
    echo "=========================================="
    echo "Step 2 完成！"
    echo ""
    echo "对比 WandB:"
    echo "  s1_prompt_new (beta=0.5) vs s2_beta10 (beta=1.0)"
    echo ""
    echo "下一步: bash scripts/run_ablation.sh --decide step2"
    echo "=========================================="
}

# ========== Step 3: 数据消融 ==========
step3() {
    local prompt_choice=$(get_decision "step1_prompt")
    local beta_choice=$(get_decision "step2_beta")
    
    if [ -z "$prompt_choice" ] || [ -z "$beta_choice" ]; then
        echo "❌ 请先完成 Step 1 和 Step 2 并做决策"
        return 1
    fi
    
    # 获取 step2 胜者的 run_name 作为 baseline
    local baseline_run
    if [ "$beta_choice" = "0.5" ]; then
        baseline_run="s1_prompt_new"
        config_path="configs/skyrl-experiments/ablation/multilevel_f05.yaml"
    else
        baseline_run="s2_beta10"
        config_path="configs/skyrl-experiments/ablation/multilevel_f1.yaml"
    fi
    
    echo "=========================================="
    echo "Step 3: 数据过滤消融"
    echo ""
    echo "使用 beta=$beta_choice"
    echo "复用 $baseline_run 作为 raw baseline"
    echo "只需跑 s3_filtered"
    echo ""
    echo "对比: $baseline_run (raw) vs s3_filtered (filtered)"
    echo "=========================================="
    
    echo "实验 3: filtered 数据"
    run_exp "s3_filtered" \
        "$config_path" \
        "$DATA_FILTERED" \
        "Step3: filtered data (compare with $baseline_run)"
    
    echo ""
    echo "=========================================="
    echo "Step 3 完成！"
    echo ""
    echo "对比 WandB:"
    echo "  $baseline_run (raw) vs s3_filtered (filtered)"
    echo ""
    echo "下一步: bash scripts/run_ablation.sh --decide step3"
    echo "=========================================="
}

# ========== Step 4: Reward 函数消融 ==========
step4() {
    local prompt_choice=$(get_decision "step1_prompt")
    local beta_choice=$(get_decision "step2_beta")
    local data_choice=$(get_decision "step3_data")
    
    if [ -z "$prompt_choice" ] || [ -z "$beta_choice" ] || [ -z "$data_choice" ]; then
        echo "❌ 请先完成 Step 1-3 并做决策"
        return 1
    fi
    
    # 获取 step3 胜者
    local baseline_run data_path
    if [ "$data_choice" = "filtered" ]; then
        baseline_run="s3_filtered"
        data_path="$DATA_FILTERED"
    else
        if [ "$beta_choice" = "0.5" ]; then
            baseline_run="s1_prompt_new"
        else
            baseline_run="s2_beta10"
        fi
        data_path="$DATA_RAW"
    fi
    
    echo "=========================================="
    echo "Step 4: Reward 函数消融"
    echo ""
    echo "使用 beta=$beta_choice, data=$data_choice"
    echo "复用 $baseline_run 作为 multilevel baseline"
    echo "只需跑 s4_hierarchical"
    echo ""
    echo "对比: $baseline_run (multilevel) vs s4_hierarchical"
    echo "=========================================="
    
    echo "实验 4: hierarchical (soft gating)"
    run_exp "s4_hierarchical" \
        "configs/skyrl-experiments/ablation/hierarchical_soft.yaml" \
        "$data_path" \
        "Step4: hierarchical (compare with $baseline_run)"
    
    echo ""
    echo "=========================================="
    echo "Step 4 完成！"
    echo ""
    echo "对比 WandB:"
    echo "  $baseline_run (multilevel) vs s4_hierarchical"
    echo ""
    echo "下一步: bash scripts/run_ablation.sh --decide step4"
    echo "=========================================="
}

# ========== 决策命令 ==========
decide() {
    local step=$1
    
    case "$step" in
        step1)
            echo "=========================================="
            echo "Step 1 决策: Prompt 版本"
            echo ""
            echo "对比 s1_prompt_old vs s1_prompt_new"
            echo "关注: localization_reward 是否 > 0"
            echo ""
            echo "预期: tool_only 胜出"
            echo "=========================================="
            read -p "输入选择 (old/tool_only): " choice
            if [ "$choice" = "old" ] || [ "$choice" = "tool_only" ]; then
                set_decision "step1_prompt" "$choice"
            else
                echo "❌ 无效选择"
            fi
            ;;
        step2)
            echo "=========================================="
            echo "Step 2 决策: Beta 值"
            echo ""
            echo "对比 s1_prompt_new (beta=0.5) vs s2_beta10 (beta=1.0)"
            echo "关注: file/module/entity reward"
            echo "=========================================="
            read -p "输入选择 (0.5/1.0): " choice
            if [ "$choice" = "0.5" ] || [ "$choice" = "1.0" ]; then
                set_decision "step2_beta" "$choice"
            else
                echo "❌ 无效选择"
            fi
            ;;
        step3)
            echo "=========================================="
            echo "Step 3 决策: 数据过滤"
            echo ""
            local beta=$(get_decision "step2_beta")
            local baseline
            if [ "$beta" = "0.5" ]; then
                baseline="s1_prompt_new"
            else
                baseline="s2_beta10"
            fi
            echo "对比 $baseline (raw) vs s3_filtered (filtered)"
            echo "关注: 收敛速度、最终 reward"
            echo "=========================================="
            read -p "输入选择 (raw/filtered): " choice
            if [ "$choice" = "raw" ] || [ "$choice" = "filtered" ]; then
                set_decision "step3_data" "$choice"
            else
                echo "❌ 无效选择"
            fi
            ;;
        step4)
            echo "=========================================="
            echo "Step 4 决策: Reward 函数"
            echo ""
            echo "对比 multilevel vs hierarchical"
            echo "关注: 各层 reward 一致性"
            echo "=========================================="
            read -p "输入选择 (multilevel/hierarchical): " choice
            if [ "$choice" = "multilevel" ] || [ "$choice" = "hierarchical" ]; then
                set_decision "step4_reward" "$choice"
            else
                echo "❌ 无效选择"
            fi
            ;;
        *)
            echo "❌ 未知步骤: $step"
            ;;
    esac
}

# ========== 状态查看 ==========
show_status() {
    echo "=========================================="
    echo "📊 Ablation V2 Status"
    echo "=========================================="
    
    echo ""
    echo "📋 决策记录:"
    if [ -f "$DECISION_FILE" ]; then
        cat "$DECISION_FILE"
    else
        echo "  (无)"
    fi
    
    echo ""
    echo "📦 实验状态:"
    
    local experiments=(
        "s1_prompt_old:Step1:Old prompt"
        "s1_prompt_new:Step1:New prompt (baseline)"
        "s2_beta10:Step2:beta=1.0"
        "s3_filtered:Step3:Filtered data"
        "s4_hierarchical:Step4:Hierarchical"
    )
    
    for entry in "${experiments[@]}"; do
        IFS=':' read -r exp_name step desc <<< "$entry"
        local exp_dir="${CKPT_BASE}/${exp_name}-${BASE_MODEL}"
        if [ -d "$exp_dir" ]; then
            local ckpts=$(ls -d "$exp_dir"/global_step_* 2>/dev/null | wc -l)
            if [ "$ckpts" -gt 0 ]; then
                local latest=$(ls -d "$exp_dir"/global_step_* | sort -t_ -k3 -rn | head -1 | xargs basename)
                echo "  ✅ $exp_name: $ckpts ckpts, $latest"
            else
                echo "  🔄 $exp_name: started"
            fi
        else
            echo "  ⏳ $exp_name: not started"
        fi
    done
    
    echo ""
    echo "=========================================="
    echo "🔄 实验流程 (贪心爬坡，无重复):"
    echo "=========================================="
    echo ""
    echo "Step 1: Prompt"
    echo "  s1_prompt_old vs s1_prompt_new"
    echo "  → $(get_decision step1_prompt)"
    echo ""
    echo "Step 2: Beta (复用 s1_prompt_new 作为 beta=0.5)"
    echo "  s1_prompt_new vs s2_beta10"
    echo "  → $(get_decision step2_beta)"
    echo ""
    echo "Step 3: Data (复用 step2 胜者作为 raw)"
    echo "  (step2胜者) vs s3_filtered"
    echo "  → $(get_decision step3_data)"
    echo ""
    echo "Step 4: Reward (复用 step3 胜者作为 multilevel)"
    echo "  (step3胜者) vs s4_hierarchical"
    echo "  → $(get_decision step4_reward)"
    echo "=========================================="
}

# ========== 单独运行某个实验 ==========
run_s1_old() {
    run_exp "s1_prompt_old" \
        "configs/skyrl-experiments/ablation/prompt_backticks_baseline.yaml" \
        "$DATA_RAW" \
        "Step1: Old prompt"
}

run_s1_new() {
    run_exp "s1_prompt_new" \
        "configs/skyrl-experiments/ablation/multilevel_f05.yaml" \
        "$DATA_RAW" \
        "Step1: New prompt (baseline)"
}

run_s2() {
    run_exp "s2_beta10" \
        "configs/skyrl-experiments/ablation/multilevel_f1.yaml" \
        "$DATA_RAW" \
        "Step2: beta=1.0"
}

# ========== 主逻辑 ==========
case "$PHASE" in
    step1|1)
        step1
        ;;
    s1_old)
        run_s1_old
        ;;
    s1_new)
        cleanup
        run_s1_new
        ;;
    step2|2)
        step2
        ;;
    s2)
        cleanup
        run_s2
        ;;
    step3|3)
        step3
        ;;
    step4|4)
        step4
        ;;
    --decide|-d)
        decide "$2"
        ;;
    --status|-s)
        show_status
        ;;
    --help|-h|help)
        echo "用法: bash scripts/run_ablation.sh <command> [--resume]"
        echo ""
        echo "贪心爬坡消融实验（无重复）"
        echo ""
        echo "选项:"
        echo "  --resume, -r  从 checkpoint 续训，自动恢复 wandb 日志"
        echo ""
        echo "实验:"
        echo "  step1      - Prompt 消融 (s1_old + s1_new)"
        echo "  s1_old     - 只跑 s1_prompt_old"
        echo "  s1_new     - 只跑 s1_prompt_new"
        echo "  step2      - Beta 消融"
        echo "  s2         - 只跑 s2_beta10"
        echo "  step3      - 数据消融"
        echo "  step4      - Reward 消融"
        echo ""
        echo "决策:"
        echo "  --decide step1/2/3/4"
        echo ""
        echo "其他:"
        echo "  --status  查看状态"
        echo ""
        echo "示例:"
        echo "  bash scripts/run_ablation.sh s1_new           # 从头开始"
        echo "  bash scripts/run_ablation.sh s1_new --resume  # 续训"
        ;;
    *)
        echo "❌ Unknown: $PHASE"
        exit 1
        ;;
esac
