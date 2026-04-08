#!/usr/bin/env python3
"""
用实测数据校准显存估算公式。

我们有两组硬数据：
1. 单卡 36 层 benchmark (无 FSDP, 无 logits):
   S=16384 → peak=22.4 GB, S=40960 → peak=33.1 GB
2. 训练 log 分析: 悬崖点在 glen ≈ 34000

从这两组数据可以精确反推每个组件的实际显存。
"""

# === 已知常量 ===
H = 2560       # Qwen3-4B hidden size
V = 151643     # vocab size
L = 36         # num layers
P = 4e9        # total params
BF16 = 2       # bytes
FSDP_SIZE = 4  # 训练卡数
GPU_MEM = 64   # GB

# === 数据点 1: 单卡 36 层 benchmark (无 FSDP, 无 lm_head) ===
# 这个 benchmark 包含: 模型参数 + grad ckpt activations + backward 临时 tensors
# 不包含: logits, FSDP 通信 buffer, optimizer
bench_data = {
    16384: 22.4,  # GB
    24576: 26.0,
    30720: 28.6,
    32768: 29.5,
    34816: 30.4,
    36864: 31.3,
    37888: 31.8,
    40960: 33.1,
}

# 线性拟合: peak = base + slope * S
# base = 模型参数 (不随 S 变化)
# slope = activations per token
import statistics
Ss = list(bench_data.keys())
peaks = list(bench_data.values())
n = len(Ss)
mean_S = statistics.mean(Ss)
mean_P = statistics.mean(peaks)
slope = sum((s - mean_S) * (p - mean_P) for s, p in zip(Ss, peaks)) / sum((s - mean_S)**2 for s in Ss)
base = mean_P - slope * mean_S

print("=== 从单卡 benchmark 拟合 ===")
print(f"peak = {base:.2f} + {slope*1e6:.2f} * S/1e6  (GB)")
print(f"base (模型参数等固定开销) = {base:.1f} GB")
print(f"slope (activations per token) = {slope*1e6:.2f} GB/M_tokens = {slope*1e3:.4f} KB/token")
print()

# 验证拟合
print("拟合验证:")
for S, actual in bench_data.items():
    predicted = base + slope * S
    print(f"  S={S:>6}: actual={actual:.1f}, predicted={predicted:.1f}, error={predicted-actual:+.1f}")

# === 数据点 2: 训练中的额外开销 ===
# 训练 = 单卡 benchmark + FSDP overhead + logits
# FSDP overhead (不随 S 变化):
#   - gradients sharded: P * BF16 / FSDP_SIZE = 2.0 GB
#   - FSDP buffers/metadata: ~1-2 GB
#   - cpu_offload backload 临时: 单层 all-gather ≈ P/L * BF16 = 0.22 GB
# 总 FSDP fixed overhead ≈ 3-4 GB

# logits (随 S 变化):
#   - lm_head 输出: S * V * BF16
#   - logprobs_from_logits_v2 的 per-row 循环:
#     peak = logits tensor + 1 row 的 fp32 log_softmax = S*V*2 + V*4
#     ≈ S * V * 2 (V*4 = 0.6 MB, 可忽略)
#   但实际上 logits tensor 在 forward 结束后一直存活到 logprobs 计算完成
#   期间 backward 还没开始，所有 grad ckpt activations 也在
#   所以 peak = activations + logits + FSDP overhead + base

print()
print("=== 训练中的完整显存估算 ===")
fsdp_overhead = 4.0  # GB, 从经验估算

print(f"训练 peak = base({base:.1f}) + fsdp_overhead({fsdp_overhead}) + slope*S + logits(S*V*BF16)")
print()
print(f"{'S':>6}  {'base':>5}  {'fsdp':>5}  {'act':>6}  {'logits':>7}  {'total':>6}  {'余量':>5}  {'状态'}")
print("-" * 65)

for S in [8000, 12000, 16000, 20000, 24000, 28000, 32000, 34000, 36000, 37000, 40960]:
    act = slope * S
    logits = S * V * BF16 / 1e9
    total = base + fsdp_overhead + act + logits
    headroom = GPU_MEM - total
    status = "正常" if headroom > 10 else ("紧张" if headroom > 5 else ("危险" if headroom > 0 else "OOM"))
    print(f"{S:>6}  {base:>5.1f}  {fsdp_overhead:>5.1f}  {act:>6.1f}  {logits:>7.1f}  {total:>6.1f}  {headroom:>5.1f}  {status}")

# 找悬崖点
print()
for S in range(20000, 50000, 500):
    total = base + fsdp_overhead + slope * S + S * V * BF16 / 1e9
    if GPU_MEM - total < 5:
        print(f"估算悬崖点 (余量 < 5 GB): S ≈ {S}")
        break

print(f"实际观察到的悬崖点: S ≈ 34000")
print()

# === 通用公式 ===
print("=== 通用估算公式 ===")
print()
print("对于任意模型在昇腾 910 (64 GB) 上做 RL 训练:")
print()
print("  peak_mem(S, SP) = model_base + fsdp_overhead + act_per_tok * S/SP + S/SP * V * 2 / 1e9")
print()
print("其中:")
print(f"  model_base ≈ total_params_GB * 1.1  (模型参数 bf16 + 少量 buffer)")
print(f"  fsdp_overhead ≈ 4 GB  (gradients shard + FSDP metadata)")
print(f"  act_per_tok ≈ L * H * 2 * 1.2 / 1e9  (grad ckpt activations, 1.2x 是 overhead)")
print(f"  logits = S/SP * V * 2 / 1e9")
print()
print("需要 SP 的条件: peak_mem(S_max, 1) > GPU_MEM - 5")
print("SP=k 时: peak_mem(S_max, k) < GPU_MEM - 5")

# 用通用公式验证我们的场景
print()
print("=== 用通用公式验证 ===")
model_base = P * BF16 / 1e9 * 1.1  # 8.8 GB
act_per_tok = L * H * BF16 * 1.2 / 1e9
print(f"Qwen3-4B: model_base={model_base:.1f}, act_per_tok={act_per_tok*1e6:.1f} KB/tok")
print(f"  (实测: base={base:.1f}, act_per_tok={slope*1e6:.1f} KB/tok)")
print(f"  误差: base {model_base-base:+.1f} GB, act_per_tok {(act_per_tok-slope)/slope*100:+.0f}%")
