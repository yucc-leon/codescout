#!/usr/bin/env python3
"""
单层 Transformer forward+backward benchmark，带显存监控。
分离 attention 和 MLP 耗时，定位 glen ~34000 处的 non-SDPA 性能悬崖。

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0 python codescout/scripts/bench_layer_cliff.py
"""
import time
import torch
import torch_npu
from transformers import AutoModelForCausalLM

DEVICE = "npu"
DTYPE = torch.bfloat16
WARMUP = 2
REPEATS = 3


def get_mem_mb():
    return torch.npu.memory_allocated() / 1e6


def bench_fn(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    mem_before = get_mem_mb()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / repeats * 1000
    peak = get_mem_mb()  # current, not peak (peak includes warmup allocs)
    peak_actual = torch.npu.max_memory_allocated() / 1e6
    return ms, peak_actual, peak_actual - mem_before


def main():
    torch.npu.set_device(0)

    model_name = "/sharedata/liyuchen/models/Qwen3-4B-Instruct-2507"
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE).to(DEVICE)
    model.train()

    # Extract components
    embed = model.model.embed_tokens
    rotary = model.model.rotary_emb
    layer0 = model.model.layers[0]

    # Freeze embed and rotary (we only want gradients through the layer)
    embed.requires_grad_(False)
    rotary.requires_grad_(False)

    B = 1
    seq_lens = [
        8192, 16384, 24576,
        28672, 30720, 32768, 33792, 34816, 35840, 36864, 37888,
        40960, 45056, 49152,
    ]

    print(f"\n{'='*100}")
    print(f"Single Decoder Layer fwd+bwd Benchmark (Qwen3-4B)")
    print(f"  B={B}, dtype={DTYPE}, warmup={WARMUP}, repeats={REPEATS}")
    print(f"{'='*100}")

    results = []
    for S in seq_lens:
        print(f"\n--- S={S} ---")
        try:
            torch.npu.empty_cache()
            torch.npu.reset_peak_memory_stats()

            input_ids = torch.randint(0, 1000, (B, S), device=DEVICE)
            position_ids = torch.arange(S, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                hidden = embed(input_ids)
                cos, sin = rotary(hidden, position_ids)

            # Detach and require grad for the layer input
            hidden = hidden.detach().requires_grad_(True)
            pos_emb = (cos.detach(), sin.detach())

            del input_ids
            torch.npu.empty_cache()

            def run_fwd_bwd():
                out = layer0(hidden, position_ids=position_ids, position_embeddings=pos_emb)
                out_tensor = out[0] if isinstance(out, tuple) else out
                out_tensor.sum().backward()
                hidden.grad = None

            ms, peak, delta = bench_fn(run_fwd_bwd)
            ms_per_tok = ms / S
            print(f"  fwd+bwd: {ms:.1f} ms ({ms_per_tok:.4f} ms/tok), peak={peak:.0f} MB, delta={delta:.0f} MB")
            results.append((S, ms, ms_per_tok, peak, delta))

        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append((S, None, None, None, None))
        finally:
            for v in ['hidden', 'pos_emb', 'position_ids', 'cos', 'sin']:
                if v in dir():
                    try: exec(f"del {v}")
                    except: pass
            torch.npu.empty_cache()

    # Summary
    print(f"\n{'='*100}")
    print(f"{'S':>6}  {'time(ms)':>10}  {'ms/tok':>8}  {'peak(MB)':>10}  {'delta(MB)':>10}  {'t_norm':>8}  {'m_norm':>8}")
    print("-" * 75)

    prev = None
    for S, ms, ms_tok, peak, delta in results:
        if ms is not None:
            t_norm = f"{ms_tok / prev[0]:.3f}" if prev else ""
            m_norm = f"{delta / prev[1]:.3f}" if prev and prev[1] > 0 else ""
            flag = " ← CLIFF" if prev and ms_tok / prev[0] > 1.15 else ""
            print(f"{S:>6}  {ms:>10.1f}  {ms_tok:>8.4f}  {peak:>10.0f}  {delta:>10.0f}  {t_norm:>8}  {m_norm:>8}{flag}")
            prev = (ms_tok, delta)
        else:
            print(f"{S:>6}  {'FAIL':>10}")
            prev = None

    print(f"{'='*100}")


if __name__ == "__main__":
    main()
