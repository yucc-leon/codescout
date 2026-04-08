#!/usr/bin/env python3
"""Benchmark chunked vs original logprobs: memory AND compute cost."""
import time
import torch
import torch_npu
import torch.nn.functional as F
import sys
sys.path.insert(0, "SkyRL/skyrl-train")
from skyrl_train.utils.torch_utils import (
    logprobs_from_logits_v2,
    chunked_logprobs_from_hidden_states,
    chunked_entropy_from_hidden_states,
    chunked_entropy_from_logits,
)

torch.npu.set_device(0)
B, H, V = 1, 2560, 151643
WARMUP, REPEATS = 3, 5

def bench(fn, label):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        fn()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / REPEATS * 1000
    peak = torch.npu.max_memory_allocated() / 1e9
    return ms, peak

print(f"{'S':>6}  {'orig_ms':>8}  {'chunk_ms':>9}  {'overhead':>9}  {'orig_GB':>8}  {'chunk_GB':>9}  {'saved_GB':>9}")
print("-" * 75)

for S in [8192, 16384, 24576, 32768, 37000, 40960]:
    hidden = torch.randn(B, S, H, dtype=torch.bfloat16, device="npu")
    lm_head = torch.nn.Linear(H, V, bias=False, dtype=torch.bfloat16, device="npu")
    labels = torch.randint(0, V, (B, S), device="npu")

    # Original: lm_head -> full logits -> logprobs
    def run_orig():
        logits = lm_head(hidden)
        lp = logprobs_from_logits_v2(logits, labels)
        return lp

    # Chunked: hidden -> chunked lm_head+logprobs
    def run_chunked():
        return chunked_logprobs_from_hidden_states(hidden, labels, lm_head)

    try:
        orig_ms, orig_gb = bench(run_orig, "orig")
    except:
        orig_ms, orig_gb = float("inf"), -1

    try:
        chunk_ms, chunk_gb = bench(run_chunked, "chunked")
    except:
        chunk_ms, chunk_gb = float("inf"), -1

    overhead = f"{chunk_ms/orig_ms:.2f}x" if orig_ms < float("inf") else "N/A"
    saved = f"{orig_gb - chunk_gb:.1f}" if orig_gb > 0 else "OOM"
    print(f"{S:>6}  {orig_ms:>8.1f}  {chunk_ms:>9.1f}  {overhead:>9}  {orig_gb:>8.2f}  {chunk_gb:>9.2f}  {saved:>9}")

    del hidden, lm_head, labels
    torch.npu.empty_cache()

print("\nNote: 'overhead' = chunked_time / original_time")
print("The extra compute cost is from re-doing lm_head matmul in chunks")
print("(less efficient than one big matmul due to smaller tile sizes)")
