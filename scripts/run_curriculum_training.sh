#!/bin/bash
# 动态课程学习训练脚本
#
# 用法:
#   ./scripts/run_curriculum_training.sh [OPTIONS]
#
# 选项:
#   --model 4b|14b          模型大小 (默认: 4b)
#   --stages 3|4            阶段数 (默认: 4)
#   --steps-per-stage N     每阶段步数 (默认: 100)
#   --config CONFIG         训练配置文件
#   --prefilter-data PATH   预过滤数据路径
#   --output-dir DIR        输出目录
#   --skip-refilter         跳过重新评估步骤
#   --dry-run               只打印命令，不执行

set -e

# 默认参数
MODEL_SIZE="4b"
NUM_STAGES=4
STEPS_PER_STAGE=100
CONFIG="configs/skyrl-experiments/ablation_v2/multilevel_f05.yaml"
PREFILTER_DATA=""
OUTPUT_DIR="output/curriculum_training"
SKIP_REFILTER=false
DRY_RUN=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --stages)
            NUM_STAGES="$2"
            shift 2
            ;;
        --steps-per-stage)
            STEPS_PER_STAGE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --prefilter-data)
            PREFILTER_DATA="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-refilter)
            SKIP_REFILTER=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 检查预过滤数据
if [ -z "$PREFILTER_DATA" ]; then
    PREFILTER_DATA="output/prefilter_9k_curriculum/train.parquet"
fi

if [ ! -f "$PREFILTER_DATA" ]; then
    echo "Error: Prefilter data not found: $PREFILTER_DATA"
    echo "Please run prefilter_data.py first."
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# 记录配置
echo "============================================================"
echo "Dynamic Curriculum Learning Training"
echo "============================================================"
echo "Model: $MODEL_SIZE"
echo "Stages: $NUM_STAGES"
echo "Steps per stage: $STEPS_PER_STAGE"
echo "Total steps: $((NUM_STAGES * STEPS_PER_STAGE))"
echo "Config: $CONFIG"
echo "Prefilter data: $PREFILTER_DATA"
echo "Output dir: $OUTPUT_DIR"
echo "Skip refilter: $SKIP_REFILTER"
echo "============================================================"

# 保存配置
cat > "$OUTPUT_DIR/curriculum_config.json" << EOF
{
    "model_size": "$MODEL_SIZE",
    "num_stages": $NUM_STAGES,
    "steps_per_stage": $STEPS_PER_STAGE,
    "total_steps": $((NUM_STAGES * STEPS_PER_STAGE)),
    "config": "$CONFIG",
    "prefilter_data": "$PREFILTER_DATA",
    "skip_refilter": $SKIP_REFILTER,
    "start_time": "$(date -Iseconds)"
}
EOF

# 当前数据路径
CURRENT_DATA="$PREFILTER_DATA"

# 执行命令的函数
run_cmd() {
    local cmd="$1"
    echo ""
    echo ">>> $cmd"
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Skipping execution"
    else
        eval "$cmd"
    fi
}

# 训练循环
for STAGE in $(seq 1 $NUM_STAGES); do
    START_STEP=$(((STAGE - 1) * STEPS_PER_STAGE))
    END_STEP=$((STAGE * STEPS_PER_STAGE))
    
    echo ""
    echo "============================================================"
    echo "Stage $STAGE / $NUM_STAGES: Steps $START_STEP - $END_STEP"
    echo "============================================================"
    
    STAGE_DATA="$OUTPUT_DIR/stage${STAGE}_data.parquet"
    STAGE_CKPT="$OUTPUT_DIR/ckpt_stage${STAGE}"
    
    # 1. 选择当前阶段的样本
    echo ""
    echo "--- Step 1: Select samples for stage $STAGE ---"
    
    # 使用渐进式选择：每阶段包含之前所有难度
    run_cmd "uv run python scripts/select_curriculum_stage.py \
        --input $CURRENT_DATA \
        --stage $STAGE \
        --progressive \
        --output $STAGE_DATA"
    
    # 2. 训练
    echo ""
    echo "--- Step 2: Train for $STEPS_PER_STAGE steps ---"
    
    if [ $STAGE -eq 1 ]; then
        # 第一阶段：从头开始
        run_cmd "uv run python src/train.py \
            --config $CONFIG \
            --data $STAGE_DATA \
            --max-steps $STEPS_PER_STAGE \
            --output $STAGE_CKPT \
            2>&1 | tee $OUTPUT_DIR/logs/stage${STAGE}_train.log"
    else
        # 后续阶段：从上一阶段 checkpoint 继续
        PREV_CKPT="$OUTPUT_DIR/ckpt_stage$((STAGE-1))"
        run_cmd "uv run python src/train.py \
            --config $CONFIG \
            --data $STAGE_DATA \
            --resume $PREV_CKPT \
            --max-steps $STEPS_PER_STAGE \
            --output $STAGE_CKPT \
            2>&1 | tee $OUTPUT_DIR/logs/stage${STAGE}_train.log"
    fi
    
    # 3. 重新评估难度（除了最后一个阶段）
    if [ $STAGE -lt $NUM_STAGES ] && [ "$SKIP_REFILTER" = false ]; then
        echo ""
        echo "--- Step 3: Re-evaluate sample difficulty ---"
        
        REFILTER_OUTPUT="$OUTPUT_DIR/refiltered_stage${STAGE}"
        
        run_cmd "uv run python scripts/prefilter_data.py \
            --model $MODEL_SIZE \
            --checkpoint $STAGE_CKPT \
            --input $PREFILTER_DATA \
            --n-samples 4 \
            --dp-size 8 \
            --output $REFILTER_OUTPUT \
            2>&1 | tee $OUTPUT_DIR/logs/stage${STAGE}_refilter.log"
        
        # 更新当前数据路径
        CURRENT_DATA="$REFILTER_OUTPUT/train.parquet"
    fi
    
    # 记录阶段完成
    echo ""
    echo "Stage $STAGE completed at $(date -Iseconds)"
    echo "{\"stage\": $STAGE, \"completed_at\": \"$(date -Iseconds)\"}" >> "$OUTPUT_DIR/progress.jsonl"
done

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo "Final checkpoint: $OUTPUT_DIR/ckpt_stage${NUM_STAGES}"
echo "Logs: $OUTPUT_DIR/logs/"
echo "============================================================"

# 更新配置
cat > "$OUTPUT_DIR/curriculum_config.json" << EOF
{
    "model_size": "$MODEL_SIZE",
    "num_stages": $NUM_STAGES,
    "steps_per_stage": $STEPS_PER_STAGE,
    "total_steps": $((NUM_STAGES * STEPS_PER_STAGE)),
    "config": "$CONFIG",
    "prefilter_data": "$PREFILTER_DATA",
    "skip_refilter": $SKIP_REFILTER,
    "start_time": "$(date -Iseconds)",
    "end_time": "$(date -Iseconds)",
    "status": "completed"
}
EOF
