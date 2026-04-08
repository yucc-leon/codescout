#!/usr/bin/env python3
"""
FSDP 多卡 benchmark，复现训练中的性能悬崖。

在 4 卡上用 FSDP2 跑 Qwen3-4B 的 forward+backward，
测不同序列长度下的 per-iteration 时间。

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        codescout/scripts/bench_fsdp_cliff.py
"""
import os
import time
import torch
import torch_npu
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoConfig

DTYPE = torch.bfloat16
WARMUP = 1
REPEATS = 2


def setup():
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    return rank


def main():
    rank = setup()
    device = f"npu:{rank}"

    model_name = "/sharedata/liyuchen/models/Qwen3-4B-Instruct-2507"
    if rank == 0:
        print(f"Loading model on {dist.get_world_size()} NPUs...")

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE).to(device)
    model.train()

    # Apply FSDP2 (fully_shard) to each layer
    mp_policy = MixedPrecisionPolicy(param_dtype=DTYPE, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    seq_lens = [
        16384, 24576, 28672, 30720, 32768, 33792, 34816, 35840, 36864, 37888, 40960,
    ]

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"FSDP2 Benchmark (Qwen3-4B, {dist.get_world_size()} NPUs)")
        print(f"  dtype={DTYPE}, grad_ckpt=True, warmup={WARMUP}, repeats={REPEATS}")
        print(f"{'='*80}")

    results = []
    for S in seq_lens:
        if rank == 0:
            print(f"\n--- S={S} ---")
        try:
            torch.npu.empty_cache()
            torch.npu.reset_peak_memory_stats()

            input_ids = torch.randint(0, 1000, (1, S), device=device)
            attention_mask = torch.ones(1, S, dtype=torch.long, device=device)
            labels = input_ids.clone()

            def run_step():
                optimizer.zero_grad()
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                out.loss.backward()
                optimizer.step()

            # Warmup
            for _ in range(WARMUP):
                run_step()
            dist.barrier()
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()

            # Benchmark
            t0 = time.perf_counter()
            for _ in range(REPEATS):
                run_step()
            torch.npu.synchronize()
            ms = (time.perf_counter() - t0) / REPEATS * 1000

            peak = torch.npu.max_memory_allocated() / 1e9
            ms_per_tok = ms / S

            if rank == 0:
                print(f"  fwd+bwd+step: {ms:.0f} ms ({ms_per_tok:.4f} ms/tok), peak={peak:.1f} GB")
            results.append((S, ms, ms_per_tok, peak))

        except Exception as e:
            if rank == 0:
                import traceback
                print(f"  FAILED: {e}")
                traceback.print_exc()
            results.append((S, None, None, None))
        finally:
            del input_ids, attention_mask, labels
            torch.npu.empty_cache()

    # Summary
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"{'S':>6}  {'time(ms)':>10}  {'ms/tok':>8}  {'peak(GB)':>10}  {'t_norm':>8}")
        print("-" * 55)
        prev_ms_tok = None
        for S, ms, ms_tok, peak in results:
            if ms is not None:
                t_norm = f"{ms_tok / prev_ms_tok:.3f}" if prev_ms_tok else ""
                flag = " ← CLIFF" if prev_ms_tok and ms_tok / prev_ms_tok > 1.15 else ""
                print(f"{S:>6}  {ms:>10.0f}  {ms_tok:>8.4f}  {peak:>10.1f}  {t_norm:>8}{flag}")
                prev_ms_tok = ms_tok
            else:
                print(f"{S:>6}  {'FAIL':>10}")
                prev_ms_tok = None
        print(f"{'='*80}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
