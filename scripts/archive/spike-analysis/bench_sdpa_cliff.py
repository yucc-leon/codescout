#!/usr/bin/env python3
"""
SDPA 性能悬崖精确定位 benchmark。

从训练 log 分析发现 glen ≈ 34000 处存在性能悬崖（ms/token 从 ~14 跳到 ~31）。
本脚本在 28000-40000 区间做细粒度测试，精确定位悬崖位置。

同时对比 SDPA vs npu_fusion_attention，看后者是否有同样的悬崖。

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0 python codescout/scripts/bench_sdpa_cliff.py
"""
import time
import torch
import torch_npu

DEVICE = "npu"
DTYPE = torch.bfloat16
WARMUP = 3
REPEATS = 5

# Qwen3-4B config
D = 128  # head_dim
H = 32   # num_heads (for attention, Qwen3-4B has 32 heads)
B = 1    # batch_size (micro_train_batch_size_per_gpu=1)


def bench_sdpa_fwd(q, k, v, warmup=WARMUP, repeats=REPEATS):
    """Benchmark SDPA forward only."""
    for _ in range(warmup):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


def bench_sdpa_fwd_bwd(q, k, v, warmup=WARMUP, repeats=REPEATS):
    """Benchmark SDPA forward + backward."""
    q_grad = q.detach().clone().requires_grad_(True)
    for _ in range(warmup):
        out = torch.nn.functional.scaled_dot_product_attention(q_grad, k, v, is_causal=True)
        out.sum().backward()
        q_grad.grad = None
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        out = torch.nn.functional.scaled_dot_product_attention(q_grad, k, v, is_causal=True)
        out.sum().backward()
        q_grad.grad = None
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


def bench_nfa_fwd(q, k, v, warmup=WARMUP, repeats=REPEATS):
    """Benchmark npu_fusion_attention forward (BNSD layout)."""
    sm_scale = 1.0 / (D ** 0.5)
    S = q.shape[2]
    for _ in range(warmup):
        torch_npu.npu_fusion_attention(
            q, k, v, H, input_layout="BNSD",
            scale=sm_scale, pre_tockens=S, next_tockens=0,
        )
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        torch_npu.npu_fusion_attention(
            q, k, v, H, input_layout="BNSD",
            scale=sm_scale, pre_tockens=S, next_tockens=0,
        )
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


def bench_nfa_fwd_bwd(q, k, v, warmup=WARMUP, repeats=REPEATS):
    """Benchmark npu_fusion_attention forward + backward (BNSD layout)."""
    sm_scale = 1.0 / (D ** 0.5)
    S = q.shape[2]
    q_grad = q.detach().clone().requires_grad_(True)
    for _ in range(warmup):
        result = torch_npu.npu_fusion_attention(
            q_grad, k, v, H, input_layout="BNSD",
            scale=sm_scale, pre_tockens=S, next_tockens=0,
        )
        out = result[0] if isinstance(result, tuple) else result
        out.sum().backward()
        q_grad.grad = None
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        result = torch_npu.npu_fusion_attention(
            q_grad, k, v, H, input_layout="BNSD",
            scale=sm_scale, pre_tockens=S, next_tockens=0,
        )
        out = result[0] if isinstance(result, tuple) else result
        out.sum().backward()
        q_grad.grad = None
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


def main():
    torch.npu.set_device(0)

    # 测试点：粗粒度覆盖全范围 + 细粒度覆盖悬崖区域
    seq_lens = [
        # 粗粒度
        8192, 12288, 16384, 20480, 24576, 28672,
        # 悬崖区域细粒度 (30000-38000, 每 1024)
        30720, 31744, 32768, 33792, 34816, 35840, 36864, 37888,
        # 超长
        40960,
    ]

    print("=" * 100)
    print(f"SDPA Performance Cliff Benchmark")
    print(f"  B={B}, H={H}, D={D}, dtype={DTYPE}, device={DEVICE}")
    print(f"  warmup={WARMUP}, repeats={REPEATS}")
    print("=" * 100)

    results = []
    for S in seq_lens:
        print(f"\n--- S={S} ---")
        try:
            q = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
            k = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
            v = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)

            sdpa_fwd = bench_sdpa_fwd(q, k, v)
            print(f"  SDPA fwd:         {sdpa_fwd:>10.1f} ms")

            sdpa_fwd_bwd = bench_sdpa_fwd_bwd(q, k, v)
            print(f"  SDPA fwd+bwd:     {sdpa_fwd_bwd:>10.1f} ms")

            nfa_fwd = bench_nfa_fwd(q, k, v)
            print(f"  NFA fwd:          {nfa_fwd:>10.1f} ms")

            nfa_fwd_bwd = bench_nfa_fwd_bwd(q, k, v)
            print(f"  NFA fwd+bwd:      {nfa_fwd_bwd:>10.1f} ms")

            results.append((S, sdpa_fwd, sdpa_fwd_bwd, nfa_fwd, nfa_fwd_bwd))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((S, None, None, None, None))
        finally:
            del q, k, v
            torch.npu.empty_cache()

    # Summary table
    print(f"\n{'=' * 100}")
    print(f"{'S':>6}  {'SDPA fwd':>10}  {'SDPA f+b':>10}  {'NFA fwd':>10}  {'NFA f+b':>10}  {'SDPA ms/tok':>12}  {'NFA ms/tok':>11}  {'ratio':>6}")
    print("-" * 100)

    prev_sdpa_ms_tok = None
    for S, sf, sfb, nf, nfb in results:
        sdpa_ms_tok = sfb / S * 1000 if sfb else 0
        nfa_ms_tok = nfb / S * 1000 if nfb else 0
        ratio = sfb / nfb if sfb and nfb else 0

        jump = ""
        if prev_sdpa_ms_tok and sdpa_ms_tok > 0:
            growth = sdpa_ms_tok / prev_sdpa_ms_tok
            if growth > 1.3:
                jump = f"  ← CLIFF {growth:.2f}x"

        print(f"{S:>6}  {sf or 0:>10.1f}  {sfb or 0:>10.1f}  {nf or 0:>10.1f}  {nfb or 0:>10.1f}  {sdpa_ms_tok:>12.4f}  {nfa_ms_tok:>11.4f}  {ratio:>6.2f}x{jump}")
        if sdpa_ms_tok > 0:
            prev_sdpa_ms_tok = sdpa_ms_tok

    # Growth rate analysis
    print(f"\n=== SDPA fwd+bwd 增长率 ===")
    for i in range(1, len(results)):
        s0, _, t0, _, _ = results[i - 1]
        s1, _, t1, _, _ = results[i]
        if t0 and t1:
            s_ratio = s1 / s0
            t_ratio = t1 / t0
            normalized = t_ratio / s_ratio  # 如果线性增长，这个值应该 ≈ 1
            flag = " ← SUPER-LINEAR" if normalized > 1.3 else ""
            print(f"  S {s0:>6} -> {s1:>6} ({s_ratio:.2f}x): time {t_ratio:.2f}x, normalized {normalized:.2f}x{flag}")

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()
