#!/usr/bin/env python3
"""
多层 Transformer forward+backward benchmark，模拟 gradient checkpointing。
测试 N 层叠加时是否在 glen ~34000 出现性能悬崖。

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0 python codescout/scripts/bench_multilayer_cliff.py
"""
import time
import torch
import torch_npu
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

DEVICE = "npu"
DTYPE = torch.bfloat16
WARMUP = 1
REPEATS = 2


def main():
    torch.npu.set_device(0)

    model_name = "/sharedata/liyuchen/models/Qwen3-4B-Instruct-2507"
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE).to(DEVICE)
    model.train()

    embed = model.model.embed_tokens
    rotary = model.model.rotary_emb
    norm = model.model.norm
    lm_head = model.lm_head
    layers = model.model.layers

    NUM_LAYERS = len(layers)  # 36
    embed.requires_grad_(False)
    rotary.requires_grad_(False)

    # Test with gradient checkpointing (like training)
    USE_GRAD_CKPT = True

    seq_lens = [
        16384, 24576, 30720, 32768, 34816, 36864, 37888, 40960,
    ]

    print(f"\n{'='*100}")
    print(f"Full {NUM_LAYERS}-Layer fwd+bwd (grad_ckpt={USE_GRAD_CKPT})")
    print(f"  dtype={DTYPE}, warmup={WARMUP}, repeats={REPEATS}")
    print(f"{'='*100}")

    results = []
    for S in seq_lens:
        print(f"\n--- S={S} ---")
        try:
            torch.npu.empty_cache()
            torch.npu.reset_peak_memory_stats()

            input_ids = torch.randint(0, 1000, (1, S), device=DEVICE)
            position_ids = torch.arange(S, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                hidden = embed(input_ids)
                cos, sin = rotary(hidden, position_ids)
            pos_emb = (cos.detach(), sin.detach())
            hidden = hidden.detach().requires_grad_(True)

            del input_ids
            torch.npu.empty_cache()

            mem_before = torch.npu.memory_allocated() / 1e9

            def run_fwd_bwd():
                h = hidden
                for i, layer in enumerate(layers):
                    if USE_GRAD_CKPT:
                        h = checkpoint(
                            layer, h,
                            None,  # attention_mask
                            position_ids,
                            None,  # past_key_values
                            False,  # use_cache
                            None,  # cache_position
                            pos_emb,  # position_embeddings
                            use_reentrant=False,
                        )
                    else:
                        h = layer(h, position_ids=position_ids, position_embeddings=pos_emb)
                    if isinstance(h, tuple):
                        h = h[0]
                h = norm(h)
                # Just sum instead of lm_head to save memory
                h.sum().backward()
                hidden.grad = None

            # Warmup
            for _ in range(WARMUP):
                run_fwd_bwd()
            torch.npu.synchronize()

            peak_warmup = torch.npu.max_memory_allocated() / 1e9
            torch.npu.reset_peak_memory_stats()

            t0 = time.perf_counter()
            for _ in range(REPEATS):
                run_fwd_bwd()
            torch.npu.synchronize()
            ms = (time.perf_counter() - t0) / REPEATS * 1000

            peak = torch.npu.max_memory_allocated() / 1e9
            ms_per_tok = ms / S

            print(f"  fwd+bwd: {ms:.0f} ms ({ms_per_tok:.4f} ms/tok)")
            print(f"  mem: before={mem_before:.1f} GB, peak={peak:.1f} GB")
            results.append((S, ms, ms_per_tok, peak))

        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append((S, None, None, None))
        finally:
            for v in ['hidden', 'pos_emb', 'position_ids', 'cos', 'sin', 'h']:
                try: exec(f"del {v}")
                except: pass
            torch.npu.empty_cache()

    # Summary
    print(f"\n{'='*100}")
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
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
