#!/usr/bin/env python3
"""
估算 RL 训练是否需要 Sequence Parallel，以及需要多大的 SP。

核心公式：
  peak_mem = logits_mem + fsdp_params + activations + optimizer
  logits_mem = S_max * V * 2 * multiplier  (multiplier=2~3, 含 log_softmax 中间结果)
  fsdp_params = total_params * 2 / fsdp_size * 2  (bf16, all-gather 时临时 2x)
  activations = S_max * H * 2 * num_layers * ckpt_ratio
  optimizer = total_params * 4 * 2 / fsdp_size  (Adam fp32, 2 states)

如果 peak_mem > 单卡显存 * 0.85 (留 15% 余量)，就需要 SP。
SP=k 时，logits_mem 和 activations 都除以 k。

Usage:
  python estimate_sp_need.py
"""

def estimate(
    model_name: str,
    total_params_B: float,  # 总参数量 (billions)
    hidden_size: int,
    vocab_size: int,
    num_layers: int,
    max_seq_len: int,       # prompt + response 的最大总长度
    num_train_gpus: int,    # 训练用的 GPU 数
    gpu_mem_gb: float,      # 单卡显存 (GB)
    param_offload: bool = False,
    optimizer_offload: bool = False,
):
    bf16 = 2  # bytes per element

    # FSDP sharding across train GPUs
    fsdp_size = num_train_gpus

    S = max_seq_len
    V = vocab_size
    H = hidden_size
    P = total_params_B * 1e9

    # 1. logits + log_softmax intermediate (最大的单项)
    # model forward 产生 (1, S, V) logits，log_softmax 再产生同样大小的中间结果
    logits_mem_gb = S * V * bf16 * 2.5 / 1e9  # 2.5x = logits + log_softmax + gather overhead

    # 2. FSDP 参数 (all-gather 时临时全量)
    params_mem_gb = P * bf16 / 1e9  # all-gather 后的全量参数

    # 3. gradient checkpointing activations
    # 每层保存 input hidden state，backward 时 recompute
    ckpt_mem_gb = num_layers * S * H * bf16 / 1e9

    # 4. optimizer states (Adam: 2x fp32 per param, sharded)
    if optimizer_offload:
        opt_mem_gb = 0
    else:
        opt_mem_gb = P * 4 * 2 / fsdp_size / 1e9

    # 5. gradients (sharded)
    grad_mem_gb = P * bf16 / fsdp_size / 1e9

    # 6. param offload 时不需要常驻参数分片
    if param_offload:
        param_shard_gb = 0
    else:
        param_shard_gb = P * bf16 / fsdp_size / 1e9

    def calc_peak(sp_size):
        # SP 把 S 维度分片，logits 和 activations 都除以 sp_size
        effective_S = S / sp_size
        l = effective_S * V * bf16 * 2.5 / 1e9
        a = num_layers * effective_S * H * bf16 / 1e9
        return l + params_mem_gb + a + opt_mem_gb + grad_mem_gb + param_shard_gb

    usable_mem = gpu_mem_gb * 0.85  # 留 15% 余量给 PyTorch allocator

    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"  params={total_params_B}B, H={H}, V={V}, layers={num_layers}")
    print(f"  max_seq_len={S}, train_gpus={num_train_gpus}, gpu_mem={gpu_mem_gb}GB")
    print(f"  param_offload={param_offload}, optimizer_offload={optimizer_offload}")
    print(f"{'='*70}")
    print(f"\n显存分解 (SP=1, 即不用 SP):")
    print(f"  logits + log_softmax:  {logits_mem_gb:>6.1f} GB")
    print(f"  FSDP params (gathered): {params_mem_gb:>6.1f} GB")
    print(f"  grad ckpt activations: {ckpt_mem_gb:>6.1f} GB")
    print(f"  optimizer states:      {opt_mem_gb:>6.1f} GB")
    print(f"  gradients (sharded):   {grad_mem_gb:>6.1f} GB")
    print(f"  param shard:           {param_shard_gb:>6.1f} GB")
    peak_no_sp = calc_peak(1)
    print(f"  ─────────────────────────────")
    print(f"  估算 peak:             {peak_no_sp:>6.1f} GB")
    print(f"  单卡可用:              {usable_mem:>6.1f} GB")
    print(f"  余量:                  {usable_mem - peak_no_sp:>6.1f} GB {'← 危险!' if usable_mem - peak_no_sp < 5 else ''}")

    print(f"\n不同 SP 下的 peak 显存:")
    print(f"  {'SP':>4}  {'peak(GB)':>10}  {'余量(GB)':>10}  {'建议':>10}")
    print(f"  {'-'*45}")
    recommended_sp = 1
    for sp in [1, 2, 4, 8]:
        if sp > num_train_gpus:
            break
        peak = calc_peak(sp)
        headroom = usable_mem - peak
        status = "✓ 安全" if headroom > 5 else ("⚠ 紧张" if headroom > 0 else "✗ OOM")
        if headroom > 5 and recommended_sp == 1:
            recommended_sp = sp
        elif headroom <= 5 and sp == 1:
            pass  # need SP
        print(f"  {sp:>4}  {peak:>10.1f}  {headroom:>10.1f}  {status:>10}")
        if recommended_sp == 1 and headroom > 5:
            recommended_sp = sp

    # Find minimum SP needed
    for sp in [1, 2, 4, 8]:
        if sp > num_train_gpus:
            break
        if calc_peak(sp) < usable_mem - 5:
            recommended_sp = sp
            break

    print(f"\n推荐: SP={recommended_sp} (DP={num_train_gpus // recommended_sp})")
    print()
    return recommended_sp


if __name__ == "__main__":
    # 我们的场景
    estimate("Qwen3-4B (我们的配置)", 4, 2560, 151643, 36, 40960, 4, 64)

    # 更大模型
    estimate("Qwen3-8B on 8 NPU", 8, 4096, 151936, 32, 34816, 8, 64)

    # 更大模型 + 更长序列
    estimate("Qwen3-8B on 8 NPU (64K)", 8, 4096, 151936, 32, 65536, 8, 64)

    # GPU 场景
    estimate("Qwen3-4B on 4 H200", 4, 2560, 151643, 36, 40960, 4, 80)

    # 小模型短序列
    estimate("Qwen2.5-7B on 8 NPU (2K)", 7, 3584, 152064, 28, 2048, 8, 64)

    # DeepSeek-V3 级别
    estimate("DeepSeek-V3 (671B MoE) on 64 NPU", 37, 7168, 129280, 61, 32768, 64, 64,
             param_offload=True, optimizer_offload=True)
