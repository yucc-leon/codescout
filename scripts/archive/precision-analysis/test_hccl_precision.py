#!/usr/bin/env python3
"""Test HCCL all-reduce and reduce-scatter numerical precision."""
import torch
import torch.distributed as dist
import torch_npu

dist.init_process_group("hccl")
rank = dist.get_rank()
ws = dist.get_world_size()
torch.npu.set_device(rank)

N = 1000000

# fp32 all-reduce
x = torch.ones(N, dtype=torch.float32, device=f"npu:{rank}") * 0.0001
expected = x.clone() * ws
dist.all_reduce(x)
diff = (x - expected).abs()
if rank == 0:
    print(f"all-reduce fp32: max_diff={diff.max().item():.2e}, mean={diff.mean().item():.2e}")

# bf16 all-reduce
y = torch.ones(N, dtype=torch.bfloat16, device=f"npu:{rank}") * 0.0001
expected_bf16 = (y.float() * ws)
dist.all_reduce(y)
diff_bf16 = (y.float() - expected_bf16).abs()
if rank == 0:
    print(f"all-reduce bf16: max_diff={diff_bf16.max().item():.2e}, mean={diff_bf16.mean().item():.2e}")

# fp32 reduce-scatter (FSDP gradient reduce)
z = torch.ones(N * ws, dtype=torch.float32, device=f"npu:{rank}") * 0.0001
out = torch.empty(N, dtype=torch.float32, device=f"npu:{rank}")
dist.reduce_scatter_tensor(out, z)
expected_rs = torch.ones(N, dtype=torch.float32, device=f"npu:{rank}") * 0.0001 * ws
diff_rs = (out - expected_rs).abs()
if rank == 0:
    print(f"reduce-scatter fp32: max_diff={diff_rs.max().item():.2e}, mean={diff_rs.mean().item():.2e}")

# Test with realistic gradient-like values (small, varied)
torch.manual_seed(42)
g = torch.randn(N, dtype=torch.float32, device=f"npu:{rank}") * 0.001
g_copy = g.clone()
dist.all_reduce(g)
g_expected = g_copy * ws  # all ranks had same seed
diff_g = (g - g_expected).abs()
if rank == 0:
    print(f"all-reduce fp32 (randn): max_diff={diff_g.max().item():.2e}, mean={diff_g.mean().item():.2e}")
    print(f"  relative max: {(diff_g / g_expected.abs().clamp(min=1e-10)).max().item():.2e}")

dist.destroy_process_group()
