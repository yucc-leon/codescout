#!/usr/bin/env python3
"""
FSDP2 多卡训练 benchmark v2。

绕过 cross_entropy OOM：不算 lm_head loss，只算 hidden states 的 sum loss。
这样可以测到长序列下 FSDP 的 forward+backward 性能。

同时加 per-phase profiling：分离 forward、backward、optimizer step 的耗时。

Usage:
    torchrun --nproc_per_node=4 codescout/scripts/bench_fsdp_cliff_v2.py
"""
import os
import time
import torch
import torch_npu
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM

DTYPE = torch.bfloat16
WARMUP = 2
REPEATS = 3


def setup():
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    return rank, dist.get_world_size()


def main():
    rank, world_size = setup()
    device = f"npu:{rank}"

    model_name = "/sharedata/liyuchen/models/Qwen3-4B-Instruct-2507"
    if rank == 0:
        print(f"Loading model on {world_size} NPUs...")

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE).to(device)
    model.train()

    # Apply FSDP2
    mp_policy = MixedPrecisionPolicy(param_dtype=DTYPE, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    # We'll skip lm_head to avoid the huge logits tensor OOM.
    # Instead, hook into the last hidden state and compute a dummy loss.
    # This still exercises all transformer layers + FSDP communication.

    seq_lens = [
        8192, 16384, 24576, 28672, 30720, 32768, 33792, 34816, 35840, 36864, 37888, 40960,
    ]

    if rank == 0:
        print(f"\n{'='*100}")
        print(f"FSDP2 Training Benchmark v2 ({world_size} NPUs, skip lm_head)")
        print(f"  dtype={DTYPE}, grad_ckpt=True, warmup={WARMUP}, repeats={REPEATS}")
        print(f"{'='*100}")

    results = []
    for S in seq_lens:
        if rank == 0:
            print(f"\n--- S={S} ---")
        try:
            torch.npu.empty_cache()
            torch.npu.reset_peak_memory_stats()

            input_ids = torch.randint(0, 1000, (1, S), device=device)
            attention_mask = torch.ones(1, S, dtype=torch.long, device=device)

            def run_step():
                optimizer.zero_grad()
                # Forward through all layers but skip lm_head
                outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs[0]  # last hidden state
                loss = hidden.sum()  # dummy loss to trigger backward
                loss.backward()
                optimizer.step()
                return loss.item()

            # Warmup
            for _ in range(WARMUP):
                run_step()
            dist.barrier()
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()

            # Benchmark with phase timing
            fwd_times = []
            bwd_times = []
            opt_times = []
            total_times = []

            for _ in range(REPEATS):
                optimizer.zero_grad()

                # Forward
                torch.npu.synchronize()
                t_fwd_start = time.perf_counter()
                outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs[0]
                loss = hidden.sum()
                torch.npu.synchronize()
                t_fwd_end = time.perf_counter()

                # Backward
                t_bwd_start = time.perf_counter()
                loss.backward()
                torch.npu.synchronize()
                t_bwd_end = time.perf_counter()

                # Optimizer step
                t_opt_start = time.perf_counter()
                optimizer.step()
                torch.npu.synchronize()
                t_opt_end = time.perf_counter()

                fwd_times.append((t_fwd_end - t_fwd_start) * 1000)
                bwd_times.append((t_bwd_end - t_bwd_start) * 1000)
                opt_times.append((t_opt_end - t_opt_start) * 1000)
                total_times.append((t_opt_end - t_fwd_start) * 1000)

            import statistics
            fwd_ms = statistics.median(fwd_times)
            bwd_ms = statistics.median(bwd_times)
            opt_ms = statistics.median(opt_times)
            total_ms = statistics.median(total_times)
            peak = torch.npu.max_memory_allocated() / 1e9

            if rank == 0:
                print(f"  total: {total_ms:.0f} ms  fwd: {fwd_ms:.0f} ms  bwd: {bwd_ms:.0f} ms  opt: {opt_ms:.0f} ms  peak: {peak:.1f} GB")
            results.append((S, total_ms, fwd_ms, bwd_ms, opt_ms, peak))

        except Exception as e:
            if rank == 0:
                import traceback
                print(f"  FAILED: {e}")
                traceback.print_exc()
            results.append((S, None, None, None, None, None))
        finally:
            del input_ids, attention_mask
            torch.npu.empty_cache()

    # Summary
    if rank == 0:
        print(f"\n{'='*100}")
        print(f"{'S':>6}  {'total':>8}  {'fwd':>8}  {'bwd':>8}  {'opt':>8}  {'peak':>6}  {'ms/tok':>8}  {'t_norm':>8}  {'bwd/fwd':>8}")
        print("-" * 85)
        prev_ms_tok = None
        for S, total, fwd, bwd, opt, peak in results:
            if total is not None:
                ms_tok = total / S
                t_norm = f"{ms_tok / prev_ms_tok:.3f}" if prev_ms_tok else ""
                flag = " ← CLIFF" if prev_ms_tok and ms_tok / prev_ms_tok > 1.15 else ""
                bwd_fwd = f"{bwd/fwd:.2f}" if fwd > 0 else ""
                print(f"{S:>6}  {total:>8.0f}  {fwd:>8.0f}  {bwd:>8.0f}  {opt:>8.0f}  {peak:>6.1f}  {ms_tok:>8.4f}  {t_norm:>8}  {bwd_fwd:>8}{flag}")
                prev_ms_tok = ms_tok
            else:
                print(f"{S:>6}  {'FAIL':>8}")
                prev_ms_tok = None
        print(f"{'='*100}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
