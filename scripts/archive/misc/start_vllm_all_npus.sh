#!/bin/bash
# Start 8 vLLM servers, one per NPU, on ports 8100-8107
set -euo pipefail

MODEL="/sharedata/liyuchen/models/Qwen3-4B-Instruct-2507"
BASE_PORT=8100
NUM_NPUS=8
PYTHON="/sharedata/liyuchen/miniforge3/envs/codescout-cann83/bin/python"

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash 2>/dev/null || true
export VLLM_ASCEND_ENABLE_NZ=0

mkdir -p /tmp/vllm_logs

for i in $(seq 0 $((NUM_NPUS - 1))); do
    PORT=$((BASE_PORT + i))
    LOG="/tmp/vllm_logs/vllm_npu${i}_port${PORT}.log"

    echo "Starting vLLM on NPU $i, port $PORT..."
    ASCEND_RT_VISIBLE_DEVICES=$i $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host 0.0.0.0 --port "$PORT" \
        --tensor-parallel-size 1 \
        --enforce-eager --dtype bfloat16 \
        --max-model-len 40960 \
        --trust-remote-code \
        --enable-auto-tool-choice --tool-call-parser hermes \
        > "$LOG" 2>&1 &

    echo "  PID=$!, log=$LOG"
done

echo ""
echo "Waiting for servers to start..."
sleep 90

echo ""
echo "Health check:"
ALL_UP=true
for i in $(seq 0 $((NUM_NPUS - 1))); do
    PORT=$((BASE_PORT + i))
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "  NPU $i (port $PORT): UP"
    else
        echo "  NPU $i (port $PORT): DOWN"
        ALL_UP=false
    fi
done

if $ALL_UP; then
    echo ""
    echo "All 8 vLLM servers are running on ports 8100-8107"
else
    echo ""
    echo "WARNING: Some servers failed to start. Check /tmp/vllm_logs/"
fi
