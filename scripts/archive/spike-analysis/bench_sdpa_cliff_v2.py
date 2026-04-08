#!/usr/bin/env python3
"""
SDPA 性能悬崖 v2: 模拟训练中的实际场景。

独立 SDPA benchmark 没有发现悬崖，但训练 log 显示 glen ~34000 处有 2x 跳变。
可能原因：
1. 训练中是完整 model forward+backward（36 层 attention + MLP），不只是单层 SDPA
2. FSDP 的 all-gather/reduce-scatter 与计算重叠
3. 显存压力导致 recomputation 或 OOM fallback

本脚本测试完整 Qwen3-4B model 的 forward+backward 在不同序列长度下的耗时。

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0 python codescout/scripts/bench_sdpa_cliff_v2.py
"""
import time
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoConfig

DEVICE = "npu"
DTYPE = torch.bfloat16
WARMUP = 2
REPEATS = 3


def bench_model_fwd_bwd(model, input_ids, attention_mask, labels, warmup=WARMUP, repeats=REPEATS):
    """Benchmark full model forward + backward."""
    for _ in range(warmup):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        model.zero_grad()
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        model.zero_grad()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


def main():
    torch.npu.set_device(0)

    model_name = "/sharedata/liyuchen/models/Qwen3-4B-Instruct-2507"
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
    ).to(DEVICE)
    model.train()
    model.gradient_checkpointing_enable()  # 训练中通常开启

    # 测试序列长度
    seq_lens = [
        8192, 16384, 24576,
        28672, 30720, 32768, 33792, 34816, 35840, 36864, 37888,
        40960,
    ]

    print(f"\n{'='*80}")
    print(f"Full Model Forward+Backward Benchmark (Qwen3-4B)")
    print(f"  dtype={DTYPE}, grad_ckpt=True, warmup={WARMUP}, repeats={REPEATS}")
    print(f"{'='*80}")

    results = []
    for S in seq_lens:
        print(f"\n--- S={S} ---")
        try:
            input_ids = torch.randint(0, 1000, (1, S), device=DEVICE)
            attention_mask = torch.ones(1, S, dtype=torch.long, device=DEVICE)
            labels = input_ids.clone()

            # Check memory before
            mem_before = torch.npu.memory_allocated() / 1e9
            
            ms = bench_model_fwd_bwd(model, input_ids, attention_mask, labels)
            
            mem_peak = torch.npu.max_memory_allocated() / 1e9
            torch.npu.reset_peak_memory_stats()
            
            print(f"  fwd+bwd: {ms:.1f} ms, mem_peak: {mem_peak:.1f} GB")
            results.append((S, ms, mem_peak))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((S, None, None))
        finally:
            del input_ids, attention_mask, labels
            torch.npu.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print(f"{'S':>6}  {'time(ms)':>10}  {'mem(GB)':>8}  {'ms/token':>10}  {'jump':>10}")
    print("-" * 55)
    prev_ms_tok = None
    for S, ms, mem in results:
        if ms:
            ms_tok = ms / S
            jump = ""
            if prev_ms_tok:
                ratio = ms_tok / prev_ms_tok
                if ratio > 1.3:
                    jump = f"← {ratio:.2f}x"
            print(f"{S:>6}  {ms:>10.1f}  {mem:>8.1f}  {ms_tok:>10.4f}  {jump:>10}")
            prev_ms_tok = ms_tok
        else:
            print(f"{S:>6}  {'FAIL':>10}  {'':>8}  {'':>10}")

    # Growth rate
    print(f"\n=== 增长率 ===")
    for i in range(1, len(results)):
        s0, t0, _ = results[i-1]
        s1, t1, _ = results[i]
        if t0 and t1:
            s_ratio = s1 / s0
            t_ratio = t1 / t0
            norm = t_ratio / s_ratio
            flag = " ← SUPER-LINEAR" if norm > 1.3 else ""
            print(f"  S {s0:>6} -> {s1:>6} ({s_ratio:.2f}x): time {t_ratio:.2f}x, norm {norm:.2f}x{flag}")


if __name__ == "__main__":
    main()
