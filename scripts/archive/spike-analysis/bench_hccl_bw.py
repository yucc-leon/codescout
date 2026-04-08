#!/usr/bin/env python3
"""
HCCL 集合通信带宽 benchmark。

测试 all-gather 和 reduce-scatter 在不同 payload 大小下的带宽，
看是否在某个大小处有性能拐点。

FSDP 的通信 payload = 单层参数量 × dtype_size。
Qwen3-4B 每层 ~111M params × 2 bytes = ~222 MB。
但 activation 的 all-reduce 大小 = seq_len × hidden_size × dtype_size。

Usage:
    torchrun --nproc_per_node=4 codescout/scripts/bench_hccl_bw.py
    torchrun --nproc_per_node=8 codescout/scripts/bench_hccl_bw.py
"""
import os
import time
import torch
import torch_npu
import torch.distributed as dist

WARMUP = 5
REPEATS = 20
DTYPE = torch.bfloat16


def setup():
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    return rank, dist.get_world_size()


def bench_all_reduce(size_mb, rank, world_size):
    """Benchmark all-reduce with given payload size."""
    n_elements = int(size_mb * 1e6 / 2)  # bf16 = 2 bytes
    tensor = torch.randn(n_elements, dtype=DTYPE, device=f"npu:{rank}")

    for _ in range(WARMUP):
        dist.all_reduce(tensor)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        dist.all_reduce(tensor)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / REPEATS * 1000

    # all-reduce: 2*(N-1)/N * size (ring algorithm)
    algo_bw = 2 * (world_size - 1) / world_size * size_mb / (ms / 1000) / 1000  # GB/s
    del tensor
    return ms, algo_bw


def bench_all_gather(size_mb, rank, world_size):
    """Benchmark all-gather (FSDP parameter gather)."""
    n_elements = int(size_mb * 1e6 / 2)
    shard = torch.randn(n_elements, dtype=DTYPE, device=f"npu:{rank}")
    output = torch.empty(n_elements * world_size, dtype=DTYPE, device=f"npu:{rank}")

    for _ in range(WARMUP):
        dist.all_gather_into_tensor(output, shard)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        dist.all_gather_into_tensor(output, shard)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / REPEATS * 1000

    total_mb = size_mb * world_size
    bw = (world_size - 1) / world_size * total_mb / (ms / 1000) / 1000  # GB/s
    del shard, output
    return ms, bw


def bench_reduce_scatter(size_mb, rank, world_size):
    """Benchmark reduce-scatter (FSDP gradient scatter)."""
    n_elements = int(size_mb * 1e6 / 2)
    input_tensor = torch.randn(n_elements * world_size, dtype=DTYPE, device=f"npu:{rank}")
    output = torch.empty(n_elements, dtype=DTYPE, device=f"npu:{rank}")

    for _ in range(WARMUP):
        dist.reduce_scatter_tensor(output, input_tensor)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        dist.reduce_scatter_tensor(output, input_tensor)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / REPEATS * 1000

    total_mb = size_mb * world_size
    bw = (world_size - 1) / world_size * total_mb / (ms / 1000) / 1000  # GB/s
    del input_tensor, output
    return ms, bw


def main():
    rank, world_size = setup()

    # Payload sizes to test (MB per rank for all-gather/reduce-scatter)
    # FSDP layer params: ~222 MB
    # Activation sizes: seq_len * hidden_size * 2 bytes
    #   S=16384: 16384 * 2560 * 2 = 80 MB
    #   S=32768: 160 MB
    #   S=37888: 185 MB
    sizes_mb = [1, 5, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]

    if rank == 0:
        print(f"{'='*90}")
        print(f"HCCL Bandwidth Benchmark ({world_size} NPUs)")
        print(f"  dtype={DTYPE}, warmup={WARMUP}, repeats={REPEATS}")
        print(f"{'='*90}")

        # All-Reduce
        print(f"\n--- All-Reduce ---")
        print(f"{'size(MB)':>10}  {'time(ms)':>10}  {'algo_bw(GB/s)':>14}")
        print("-" * 40)

    for size in sizes_mb:
        try:
            ms, bw = bench_all_reduce(size, rank, world_size)
            if rank == 0:
                print(f"{size:>10}  {ms:>10.2f}  {bw:>14.2f}")
        except Exception as e:
            if rank == 0:
                print(f"{size:>10}  FAILED: {e}")
        torch.npu.empty_cache()

    dist.barrier()

    if rank == 0:
        print(f"\n--- All-Gather (FSDP param gather) ---")
        print(f"{'shard(MB)':>10}  {'total(MB)':>10}  {'time(ms)':>10}  {'bw(GB/s)':>10}")
        print("-" * 50)

    for size in sizes_mb:
        try:
            ms, bw = bench_all_gather(size, rank, world_size)
            if rank == 0:
                print(f"{size:>10}  {size*world_size:>10}  {ms:>10.2f}  {bw:>10.2f}")
        except Exception as e:
            if rank == 0:
                print(f"{size:>10}  FAILED: {e}")
        torch.npu.empty_cache()

    dist.barrier()

    if rank == 0:
        print(f"\n--- Reduce-Scatter (FSDP grad scatter) ---")
        print(f"{'shard(MB)':>10}  {'total(MB)':>10}  {'time(ms)':>10}  {'bw(GB/s)':>10}")
        print("-" * 50)

    for size in sizes_mb:
        try:
            ms, bw = bench_reduce_scatter(size, rank, world_size)
            if rank == 0:
                print(f"{size:>10}  {size*world_size:>10}  {ms:>10.2f}  {bw:>10.2f}")
        except Exception as e:
            if rank == 0:
                print(f"{size:>10}  FAILED: {e}")
        torch.npu.empty_cache()

    dist.barrier()
    if rank == 0:
        print(f"\n{'='*90}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
