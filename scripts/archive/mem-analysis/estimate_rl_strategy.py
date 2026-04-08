#!/usr/bin/env python3
"""
估算不同模型在昇腾 910 (64GB) 上做 RL 训练的并行策略。

核心约束：
1. 单卡显存 64 GB，可用 ~54 GB (留 15% 给 allocator)
2. RL 训练的显存杀手是 logits tensor: S * V * 2 bytes
3. FSDP 分片参数和 optimizer，但 logits 和 activations 不分片
4. SP 把 S 维度分片，logits 和 activations 都除以 SP

简化假设：
- 使用 FSDP2 + gradient checkpointing + cpu_offload (optimizer)
- micro_batch_size = 1
- bf16 训练
- MoE 模型只算 active params (每 token 激活的参数量)
"""

models = [
    # (name, total_params_B, active_params_B, hidden, vocab, layers, is_moe)
    ("Qwen3-1.7B",     1.7,  1.7,  2048, 151936, 28, False),
    ("Qwen3-4B",       4.0,  4.0,  2560, 151936, 36, False),
    ("Qwen3-8B",       8.0,  8.0,  4096, 151936, 36, False),
    ("Qwen3-14B",     14.0, 14.0,  5120, 151936, 48, False),
    ("Qwen3-30B-MoE", 30.0,  3.3,  2048, 151936, 48, True),
    ("Qwen3-32B",     32.0, 32.0,  5120, 151936, 64, False),
    ("Qwen2.5-72B",   72.0, 72.0,  8192, 152064, 80, False),
    ("DS-V3-120B-MoE",120.0, 12.0, 5120, 129280, 60, True),  # 估算
    ("DS-V3-235B-MoE",235.0, 22.0, 7168, 129280, 61, True),  # 估算
    ("DS-V3-671B-MoE",671.0, 37.0, 7168, 129280, 61, True),
    ("1T-MoE",       1000.0, 50.0, 8192, 152000, 80, True),   # 假设
]

# 常见 RL 训练的 max_seq_len 配置
seq_configs = [8192, 16384, 32768, 49152]

GPU_MEM_GB = 64
USABLE_MEM_GB = 54  # 85% of 64
BF16 = 2
FP32 = 4


def estimate_peak(active_P_B, total_P_B, H, V, L, S, num_gpus, sp_size):
    """估算单卡 peak 显存 (GB)。"""
    P_active = active_P_B * 1e9
    P_total = total_P_B * 1e9
    fsdp_size = num_gpus  # 假设全分片
    S_eff = S / sp_size

    # logits: lm_head 输出 (1, S_eff, V) + log_softmax 中间结果
    # 保守估计 2x (logits + 1 copy for log_softmax)
    logits = S_eff * V * BF16 * 2 / 1e9

    # FSDP all-gather: 逐层做，peak 是单层全量参数
    layer_params = P_total / L
    fsdp_gather = layer_params * BF16 / 1e9  # 单层 all-gather

    # gradient checkpointing activations: 每层保留 input hidden state
    # 实际上 HF 默认每层 checkpoint，保留所有层的 input
    activations = L * S_eff * H * BF16 / 1e9

    # optimizer: cpu_offload 假设为 True (大模型基本都需要)
    optimizer = 0

    # gradients (sharded)
    gradients = P_total * BF16 / fsdp_size / 1e9

    # 参数 shard (cpu_offload 时 backload 的临时开销 ≈ 全量参数)
    # 因为 FSDP forward 时逐层 all-gather，peak 是单层
    param_shard = fsdp_gather  # 已经算了

    # 其他 buffer (FSDP internal, HCCL, etc.)
    buffer = 2.0

    peak = logits + fsdp_gather + activations + gradients + buffer
    return peak, logits, activations, gradients


def min_gpus_for_params(total_P_B):
    """最少需要多少卡才能放下参数 (FSDP sharded, bf16)。"""
    param_mem = total_P_B * 1e9 * BF16 / 1e9  # GB, 全量
    # FSDP sharded + gradients sharded + optimizer (cpu offload)
    # 每卡需要: param_shard + grad_shard = 2 * P * bf16 / N
    # 加上 all-gather 临时开销 (单层全量)
    # 简化: 每卡至少需要 param_shard < 可用显存的 30%
    per_gpu_budget = USABLE_MEM_GB * 0.3
    return max(1, int((param_mem / per_gpu_budget) + 0.99))


print(f"{'Model':<20} {'S':>6} {'GPUs':>5} {'SP':>3} {'DP':>3} {'peak':>6} {'logits':>7} {'act':>5} {'余量':>5} {'策略'}")
print("=" * 95)

for name, total_P, active_P, H, V, L, is_moe in models:
    for S in seq_configs:
        # 确定最少 GPU 数
        min_gpus = min_gpus_for_params(total_P)
        # 取 8 的倍数 (一台机器 8 卡)
        num_gpus = max(8, ((min_gpus + 7) // 8) * 8)
        # 训练卡数 = 总卡数的一半 (另一半做推理)
        train_gpus = num_gpus // 2

        # 尝试不同 SP
        best_sp = None
        for sp in [1, 2, 4, 8, 16]:
            if sp > train_gpus:
                break
            peak, logits, act, grad = estimate_peak(active_P, total_P, H, V, L, S, train_gpus, sp)
            if peak < USABLE_MEM_GB:
                best_sp = sp
                best_peak = peak
                best_logits = logits
                best_act = act
                break

        if best_sp is None:
            # 即使 SP=max 也不够，需要更多卡或 TP
            strategy = "需要 TP+SP 或更多卡"
            print(f"{name:<20} {S:>6} {num_gpus:>5} {'?':>3} {'?':>3} {'>54':>6} {'':>7} {'':>5} {'<0':>5} {strategy}")
        else:
            dp = train_gpus // best_sp
            headroom = USABLE_MEM_GB - best_peak
            if best_sp == 1:
                strategy = "FSDP only"
            else:
                strategy = f"FSDP+SP{best_sp}"
            if is_moe:
                strategy += " (MoE)"
            print(f"{name:<20} {S:>6} {num_gpus:>5} {best_sp:>3} {dp:>3} {best_peak:>5.0f}G {best_logits:>5.1f}G {best_act:>4.0f}G {headroom:>4.0f}G  {strategy}")

print()
print("注意：")
print("1. 这是粗略估算，实际显存可能偏差 ±20%")
print("2. MoE 模型的 active params 是估算值")
print("3. 假设 optimizer cpu_offload=True, gradient_checkpointing=True")
print("4. GPU 数 = 训练卡 + 推理卡 (各一半)")
print("5. 悬崖点可能比估算更早出现 (FSDP allocator overhead)")
print("6. 实际部署时建议先跑几个 step 验证，观察 per-iteration 时间是否有突变")
