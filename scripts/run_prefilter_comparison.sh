#!/bin/bash
# 预过滤对比实验
# 对比不同配置在样本上的 reward 分布

set -e

MODEL="${1:-4b}"
MAX_SAMPLES="${2:-100}"
N_SAMPLES="${3:-8}"
OUTPUT_BASE="output/prefilter_ablation_v2_${MODEL}"

echo "=============================================="
echo "预过滤对比实验 - Ablation V2"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Max samples: ${MAX_SAMPLES}"
echo "N-samples per instance: ${N_SAMPLES}"
echo "Output base: ${OUTPUT_BASE}"
echo "=============================================="

# Ablation V2 配置列表
declare -A CONFIGS=(
    # Baseline
    ["v2_multilevel_f05"]="configs/skyrl-experiments/ablation_v2/multilevel_f05.yaml"
    
    # Beta 对比
    ["v2_multilevel_f1"]="configs/skyrl-experiments/ablation_v2/multilevel_f1.yaml"
    
    # Reward 函数对比
    ["v2_hierarchical_soft"]="configs/skyrl-experiments/ablation_v2/hierarchical_soft.yaml"
    ["v2_hierarchical_no_bonus"]="configs/skyrl-experiments/ablation_v2/hierarchical_no_bonus.yaml"
    
    # Format reward 对比
    ["v2_multilevel_no_format"]="configs/skyrl-experiments/ablation_v2/multilevel_no_format.yaml"
)

# 运行每个配置
for exp_name in "${!CONFIGS[@]}"; do
    config_path="${CONFIGS[$exp_name]}"
    output_dir="${OUTPUT_BASE}/${exp_name}"
    
    echo ""
    echo "=============================================="
    echo "Running: ${exp_name}"
    echo "Config: ${config_path}"
    echo "Output: ${output_dir}"
    echo "=============================================="
    
    # 检查是否已完成
    if [ -f "${output_dir}/stats.json" ]; then
        echo "⏭️  Skipping ${exp_name}: already completed"
        cat "${output_dir}/stats.json"
        continue
    fi
    
    uv run python scripts/prefilter_data.py \
        --model "${MODEL}" \
        --config "${config_path}" \
        --max-samples "${MAX_SAMPLES}" \
        --n-samples "${N_SAMPLES}" \
        --output "${output_dir}" \
        --resume
    
    echo "✅ Completed: ${exp_name}"
done

echo ""
echo "=============================================="
echo "📊 Summary"
echo "=============================================="

# 汇总结果
for exp_name in "${!CONFIGS[@]}"; do
    output_dir="${OUTPUT_BASE}/${exp_name}"
    if [ -f "${output_dir}/stats.json" ]; then
        echo ""
        echo "--- ${exp_name} ---"
        cat "${output_dir}/stats.json" | python -c "
import sys, json
d = json.load(sys.stdin)
total = d['total_rollouts']
tool_rate = 100*d['called_finish_tool']/total if total > 0 else 0
print(f\"  Total: {total}\")
print(f\"  Called finish tool: {d['called_finish_tool']} ({tool_rate:.1f}%)\")
print(f\"  Avg reward: {d['avg_reward']:.3f}\")
print(f\"  Max reward: {d['max_reward']:.3f}\")
print(f\"  Min reward: {d['min_reward']:.3f}\")
print(f\"  Distribution: {d['reward_distribution']}\")
"
    fi
done

echo ""
echo "=============================================="
echo "✅ All experiments completed"
echo "=============================================="
