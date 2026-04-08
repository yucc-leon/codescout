#!/usr/bin/env python3
"""用实际数据校准显存估算模型。"""

V = 151643
H = 2560
bf16 = 2

# 从 benchmark 反推 activations per token:
# 单卡 36 层 + grad ckpt, S=16384 时 peak=22.4 GB
# 模型参数 = 8 GB
# activations(16384) = 22.4 - 8 = 14.4 GB
# per token = 14.4 GB / 16384 = 879 KB/token
act_per_token_kb = 879

# 训练中的 fixed 开销 (FSDP shard + gradients + buffers)
# cpu_offload=true: 参数 backload 时临时 ~8 GB, optimizer=0
# gradients sharded: 2 GB, buffers: ~2 GB
fixed_gb = 12

print("S         logits    act      fixed    peak     vs 64GB")
print("-" * 60)
for S in [8000, 12000, 16000, 20000, 24000, 28000, 32000, 34000, 36000, 37000, 40960]:
    logits_gb = S * V * bf16 / 1e9
    act_gb = S * act_per_token_kb / 1e6
    peak = fixed_gb + act_gb + logits_gb
    headroom = 64 - peak
    flag = "SPIKE" if headroom < 5 else ""
    print(f"S={S:>6}  {logits_gb:>6.1f}GB  {act_gb:>5.1f}GB  {fixed_gb}GB  {peak:>5.1f}GB  余量{headroom:>5.1f}GB  {flag}")

print()
print("悬崖点（余量 < 5 GB）在 S ≈", end=" ")
for S in range(20000, 45000, 1000):
    peak = fixed_gb + S * act_per_token_kb / 1e6 + S * V * bf16 / 1e9
    if 64 - peak < 5:
        print(f"{S}")
        break

print()
print("实际观察到的悬崖点: S ≈ 34000")
print("估算的悬崖点和实际是否吻合？")
