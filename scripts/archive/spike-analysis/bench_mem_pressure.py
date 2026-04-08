#!/usr/bin/env python3
"""
测试显存压力对计算性能的影响。

假设：当 NPU 显存使用率高时，分配器需要做 defragmentation，导致计算变慢。
方法：先分配一大块显存（模拟 FSDP 的参数 + optimizer），然后测不同序列长度的计算。

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0 python codescout/scripts/bench_mem_pressure.py
"""
import time
import torch
import torch_npu
from transformers import AutoModelForCausalLM

DEVICE = "npu"
DTYPE = torch.bfloat16


def bench_layer(layer, embed, rotary, S, warmup=2, repeats=3):
    """Benchmark single layer fwd+bwd at sequence length S."""
    input_ids = torch.randint(0, 1000, (1, S), device=DEVICE)
    position_ids = torch.arange(S, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        hidden = embed(input_ids)
        cos, sin = rotary(hidden, position_ids)
    pos_emb = (cos.detach(), sin.detach())
    hidden = hidden.detach().requires_grad_(True)
    del input_ids

    def run():
        out = layer(hidden, position_ids=position_ids, position_embeddings=pos_emb)
        out_tensor = out[0] if isinstance(out, tuple) else out
        out_tensor.sum().backward()
        hidden.grad = None

    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        run()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / repeats * 1000

    del hidden, pos_emb, position_ids
    torch.npu.empty_cache()
    return ms


def main():
    torch.npu.set_device(0)

    model_name = "/sharedata/liyuchen/models/Qwen3-4B-Instruct-2507"
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE).to(DEVICE)
    model.train()

    embed = model.model.embed_tokens
    rotary = model.model.rotary_emb
    layer0 = model.model.layers[0]
    embed.requires_grad_(False)
    rotary.requires_grad_(False)

    total_mem = torch.npu.get_device_properties(0).total_memory / 1e9
    model_mem = torch.npu.memory_allocated() / 1e9
    print(f"Total NPU memory: {total_mem:.1f} GB")
    print(f"Model memory: {model_mem:.1f} GB")

    seq_lens = [16384, 32768, 36864, 40960]

    # Test 1: No extra memory pressure
    print(f"\n=== Test 1: No extra memory pressure ===")
    for S in seq_lens:
        ms = bench_layer(layer0, embed, rotary, S)
        mem = torch.npu.memory_allocated() / 1e9
        print(f"  S={S:>6}: {ms:>8.1f} ms, mem={mem:.1f} GB")

    # Test 2: Allocate extra memory to simulate FSDP optimizer states
    pressure_sizes = [10, 20, 30, 40, 50]
    for pressure_gb in pressure_sizes:
        print(f"\n=== Test 2: +{pressure_gb} GB memory pressure ===")
        try:
            # Allocate dummy tensors
            n_elements = int(pressure_gb * 1e9 / 2)  # bf16 = 2 bytes
            dummy = torch.zeros(n_elements, dtype=DTYPE, device=DEVICE)
            mem_after = torch.npu.memory_allocated() / 1e9
            free = total_mem - mem_after
            print(f"  Allocated: {mem_after:.1f} GB, Free: {free:.1f} GB")

            for S in seq_lens:
                try:
                    ms = bench_layer(layer0, embed, rotary, S)
                    print(f"  S={S:>6}: {ms:>8.1f} ms")
                except Exception as e:
                    print(f"  S={S:>6}: FAILED ({e})")

            del dummy
            torch.npu.empty_cache()
        except Exception as e:
            print(f"  FAILED to allocate: {e}")
            break


if __name__ == "__main__":
    main()
