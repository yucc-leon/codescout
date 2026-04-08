#!/usr/bin/env python3
"""
分析 FSDP overhead：理论通信时间 vs 实际训练时间。

已知数据：
- 单卡 36 层 fwd+bwd benchmark（无通信）
- HCCL 带宽 benchmark（all-gather ~175 GB/s, reduce-scatter ~150 GB/s）
- 训练 log 中的 per-iteration 时间

计算：
- 理论 FSDP 通信时间 = 36 层 × (all-gather + reduce-scatter) × 参数量/层
- 理论总时间 = 计算时间 + 通信时间（假设无 overlap）
- 实际总时间 = 训练 log 中的 per-iteration 时间
- 如果 实际 >> 理论，说明有额外 overhead
"""
import re
import statistics

# Qwen3-4B config
NUM_LAYERS = 36
HIDDEN = 2560
INTERMEDIATE = 9728
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = HIDDEN // NUM_HEADS  # 80

# Per-layer parameter count (approximate)
# Self-attention: q_proj + k_proj + v_proj + o_proj
#   q: hidden * hidden = 2560 * 2560 = 6.5M
#   k: hidden * (kv_heads * head_dim) = 2560 * 640 = 1.6M
#   v: same as k = 1.6M
#   o: hidden * hidden = 6.5M
# MLP: gate_proj + up_proj + down_proj
#   gate: hidden * intermediate = 2560 * 9728 = 24.9M
#   up: same = 24.9M
#   down: intermediate * hidden = 24.9M
# LayerNorm: 2 * hidden = 5120 (negligible)
# Total per layer: ~91M params

ATTN_PARAMS = HIDDEN * HIDDEN + 2 * HIDDEN * (NUM_KV_HEADS * HEAD_DIM) + HIDDEN * HIDDEN
MLP_PARAMS = 3 * HIDDEN * INTERMEDIATE
LAYER_PARAMS = ATTN_PARAMS + MLP_PARAMS
TOTAL_PARAMS = LAYER_PARAMS * NUM_LAYERS

print(f"=== Qwen3-4B Parameter Count ===")
print(f"  Attention per layer: {ATTN_PARAMS/1e6:.1f}M")
print(f"  MLP per layer: {MLP_PARAMS/1e6:.1f}M")
print(f"  Total per layer: {LAYER_PARAMS/1e6:.1f}M")
print(f"  Total model: {TOTAL_PARAMS/1e6:.1f}M ({TOTAL_PARAMS*2/1e9:.2f} GB in bf16)")

# FSDP communication per layer (4 GPUs):
# Forward: all-gather params (layer_params * 2 bytes)
# Backward: all-gather params (for recompute) + reduce-scatter grads
# With reshard_after_forward=True:
#   Forward: 1 all-gather per layer
#   Backward: 1 all-gather (recompute) + 1 reduce-scatter per layer
LAYER_SIZE_MB = LAYER_PARAMS * 2 / 1e6  # bf16
WORLD_SIZE = 4

# From HCCL benchmark (4 NPUs):
# all-gather: shard=50MB -> 0.89ms, shard=100MB -> 1.69ms, shard=200MB -> 3.28ms
# reduce-scatter: shard=50MB -> 1.10ms, shard=100MB -> 2.08ms, shard=200MB -> 3.67ms
# Layer shard size = LAYER_SIZE_MB / WORLD_SIZE
SHARD_SIZE_MB = LAYER_SIZE_MB / WORLD_SIZE

# Interpolate from benchmark data
AG_BW_GBS = 175  # GB/s (all-gather bandwidth at large payloads)
RS_BW_GBS = 150  # GB/s (reduce-scatter bandwidth)

# all-gather time for one layer
AG_TIME_MS = LAYER_SIZE_MB / (AG_BW_GBS * 1000) * 1000  # total_size / bandwidth
RS_TIME_MS = LAYER_SIZE_MB / (RS_BW_GBS * 1000) * 1000

print(f"\n=== FSDP Communication per Layer ===")
print(f"  Layer size: {LAYER_SIZE_MB:.1f} MB (bf16)")
print(f"  Shard size: {SHARD_SIZE_MB:.1f} MB (per GPU)")
print(f"  All-gather time: {AG_TIME_MS:.2f} ms")
print(f"  Reduce-scatter time: {RS_TIME_MS:.2f} ms")

# Total communication per training step (forward + backward)
# Forward: 36 all-gathers
# Backward: 36 all-gathers (recompute with grad_ckpt) + 36 reduce-scatters
# With gradient checkpointing: backward does recompute, so 2x all-gather
TOTAL_AG_FWD = NUM_LAYERS * AG_TIME_MS
TOTAL_AG_BWD = NUM_LAYERS * AG_TIME_MS  # recompute
TOTAL_RS_BWD = NUM_LAYERS * RS_TIME_MS
TOTAL_COMM_MS = TOTAL_AG_FWD + TOTAL_AG_BWD + TOTAL_RS_BWD

print(f"\n=== Total Communication per Micro-batch (no overlap) ===")
print(f"  Forward all-gather: {TOTAL_AG_FWD:.1f} ms")
print(f"  Backward all-gather (recompute): {TOTAL_AG_BWD:.1f} ms")
print(f"  Backward reduce-scatter: {TOTAL_RS_BWD:.1f} ms")
print(f"  Total: {TOTAL_COMM_MS:.1f} ms")

# Single-card compute benchmark results
BENCH_COMPUTE = {
    16384: 5021,
    24576: 9231,
    30720: 12989,
    32768: 14411,
    34816: 15911,
    36864: 17358,
    37888: 18099,
    40960: 20577,
}

# Training log per-iteration times (from analyze_cliff_v2.py)
LOG = "/sharedata/liyuchen/workspace/logs/0401_1643_Qwen3-4B-Instruct-2507-npu-8x8.log"
train_data = []
with open(LOG) as f:
    for line in f:
        if "Policy Train epoch" not in line or "100%" not in line:
            continue
        m_time = re.search(r"(\d+\.\d+)s/it", line)
        m_glen = re.search(r"glen=([0-9.e+]+)", line)
        if m_time and m_glen:
            glen = float(m_glen.group(1))
            per_it = float(m_time.group(1))
            train_data.append((glen, per_it * 1000))  # convert to ms

# Deduplicate
train_unique = train_data[::2]

def interp_compute(glen):
    keys = sorted(BENCH_COMPUTE.keys())
    if glen <= keys[0]:
        return BENCH_COMPUTE[keys[0]] * glen / keys[0]
    if glen >= keys[-1]:
        return BENCH_COMPUTE[keys[-1]] * glen / keys[-1]
    for i in range(len(keys) - 1):
        if keys[i] <= glen <= keys[i+1]:
            frac = (glen - keys[i]) / (keys[i+1] - keys[i])
            return BENCH_COMPUTE[keys[i]] + frac * (BENCH_COMPUTE[keys[i+1]] - BENCH_COMPUTE[keys[i]])
    return 0

print(f"\n=== Training vs Theory ===")
print(f"{'glen':>8}  {'compute':>10}  {'comm':>8}  {'theory':>10}  {'actual':>10}  {'overhead':>10}  {'ratio':>6}")
print("-" * 75)

buckets = {}
for glen, actual_ms in train_unique:
    b = int(glen // 5000) * 5000
    compute_ms = interp_compute(glen)
    theory_ms = compute_ms + TOTAL_COMM_MS
    overhead_ms = actual_ms - theory_ms
    ratio = actual_ms / theory_ms if theory_ms > 0 else 0
    buckets.setdefault(b, []).append((glen, compute_ms, theory_ms, actual_ms, overhead_ms, ratio))

for b in sorted(buckets.keys()):
    items = buckets[b]
    mean_compute = statistics.mean([x[1] for x in items])
    mean_theory = statistics.mean([x[2] for x in items])
    mean_actual = statistics.mean([x[3] for x in items])
    mean_overhead = statistics.mean([x[4] for x in items])
    mean_ratio = statistics.mean([x[5] for x in items])
    n = len(items)
    print(f"{b:>6}-{b+5000:>5}  {mean_compute:>10.0f}  {TOTAL_COMM_MS:>8.0f}  {mean_theory:>10.0f}  {mean_actual:>10.0f}  {mean_overhead:>10.0f}  {mean_ratio:>6.2f}x  (n={n})")

print(f"\n=== 关键发现 ===")
print(f"理论通信时间（无 overlap）: {TOTAL_COMM_MS:.0f} ms")
print(f"如果通信和计算完全 overlap，overhead 应该 ≈ 0")
print(f"如果通信完全串行，overhead 应该 ≈ {TOTAL_COMM_MS:.0f} ms")
print(f"实际 overhead 如果远超 {TOTAL_COMM_MS:.0f} ms，说明有其他因素")
