#!/usr/bin/env python3
"""
批量测试不同模型 size 在昇腾上做 RL 训练的显存需求。

不需要模型权重——用随机初始化的单层 transformer + lm_head 测量实际显存，
然后组合成 FSDP 训练的 peak 估算。

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0 python rl_mem_survey.py
"""
import gc
import sys
import torch

if hasattr(torch, "npu") and torch.npu.is_available():
    DEVICE_TYPE = "npu"
    dev = torch.npu
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
    dev = torch.cuda
else:
    sys.exit("No GPU/NPU found.")

dev.set_device(0)
TOTAL_MEM_GB = dev.get_device_properties(0).total_memory / 1e9
BF16 = 2


def empty_cache():
    gc.collect()
    dev.empty_cache()


def measure_layer_act(H, intermediate_size, num_kv_heads, num_heads, head_dim, S, device="npu:0"):
    """Measure single-layer fwd+bwd activation memory using a minimal transformer layer."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    cfg = Qwen3Config(
        hidden_size=H,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_hidden_layers=1,
        max_position_embeddings=65536,
        _attn_implementation="sdpa",
    )
    layer = Qwen3DecoderLayer(cfg, layer_idx=0).to(dtype=torch.bfloat16, device=device)
    layer.train()

    # Build rotary embeddings
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    rotary = Qwen3RotaryEmbedding(config=cfg).to(device)

    hidden = torch.randn(1, S, H, dtype=torch.bfloat16, device=device, requires_grad=True)
    pos_ids = torch.arange(S, device=device).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rotary(hidden, pos_ids)
    pos_emb = (cos.detach(), sin.detach())

    empty_cache()
    dev.reset_peak_memory_stats()
    mem_before = dev.memory_allocated() / 1e9

    out = layer(hidden, position_ids=pos_ids, position_embeddings=pos_emb)
    out_t = out[0] if isinstance(out, tuple) else out
    out_t.sum().backward()
    dev.synchronize()

    peak = dev.max_memory_allocated() / 1e9
    act_mem = peak - mem_before

    del layer, rotary, hidden, pos_ids, pos_emb, out, out_t
    empty_cache()
    return act_mem


def measure_logits_mem(H, V, S, device="npu:0"):
    """Measure lm_head + logprobs computation memory."""
    lm_head = torch.nn.Linear(H, V, bias=False, dtype=torch.bfloat16, device=device)
    hidden = torch.randn(1, S, H, dtype=torch.bfloat16, device=device)
    labels = torch.randint(0, V, (1, S), device=device)

    empty_cache()
    dev.reset_peak_memory_stats()
    mem_before = dev.memory_allocated() / 1e9

    logits = lm_head(hidden)
    # Per-row fp32 log_softmax (matches SkyRL _v2)
    lp_list = []
    for row_l, row_lab in zip(logits, labels):
        row_lp = torch.nn.functional.log_softmax(row_l.float(), dim=-1)
        lp_list.append(row_lp.gather(-1, row_lab.unsqueeze(-1)).squeeze(-1))
    log_probs = torch.stack(lp_list)
    loss = log_probs.sum()
    loss.backward()
    dev.synchronize()

    peak = dev.max_memory_allocated() / 1e9
    logits_mem = peak - mem_before

    del lm_head, hidden, labels, logits, log_probs, loss
    empty_cache()
    return logits_mem


# Model configs: (name, total_params_B, H, intermediate, num_heads, num_kv_heads, head_dim, V, L)
MODELS = [
    ("Qwen3-1.7B",      1.7,  2048,  11008, 16,  4, 128, 151936, 28),
    ("Qwen3-4B",        4.0,  2560,  18944, 32,  8, 128, 151936, 36),
    ("Qwen3-8B",        8.0,  4096,  12288, 32,  8, 128, 151936, 36),
    ("Qwen3-14B",      14.0,  5120,  17408, 40,  8, 128, 151936, 48),
    ("Qwen3-32B",      32.0,  5120,  25600, 64,  8, 128, 151936, 64),
    # MoE: use active params for compute, total params for FSDP sharding
    ("Qwen3-30B-MoE",  30.0,  2048,  10240, 16,  4, 128, 151936, 48),  # active ~3.3B
    # Larger models - may need to test with smaller S
    ("Qwen2.5-72B",    72.0,  8192,  29568, 64,  8, 128, 152064, 80),
]

# Sequence lengths to test
SEQ_LENS = [8192, 16384, 32768, 40960]

# Common training configs
TRAIN_CONFIGS = [
    # (train_gpus, label)
    (4, "4卡"),
    (8, "8卡"),
]


def estimate_peak(layer_act, logits_mem, L, total_params_B, H, S, train_gpus, cpu_offload=True):
    """Estimate FSDP training peak memory."""
    P = total_params_B * 1e9
    # Checkpoint saved activations: L layers * (B, S, H) bf16
    ckpt_saved = L * S * H * BF16 / 1e9
    # One layer recompute (measured)
    total_act = ckpt_saved + layer_act
    # FSDP overhead
    param_shard = P * BF16 / train_gpus / 1e9
    grad_shard = P * BF16 / train_gpus / 1e9
    allgather_buf = (P / L) * BF16 / 1e9
    opt = 0 if cpu_offload else P * 4 * 2 / train_gpus / 1e9
    fsdp = param_shard + grad_shard + allgather_buf + opt + 1.0

    peak = total_act + logits_mem + fsdp
    return peak, total_act, logits_mem, fsdp


def main():
    print(f"Device: {DEVICE_TYPE} ({TOTAL_MEM_GB:.0f} GB)")
    print(f"Testing {len(MODELS)} models × {len(SEQ_LENS)} seq_lens")
    print()

    # Measure all models
    all_results = {}
    for name, total_P, H, inter, nh, nkv, hd, V, L in MODELS:
        print(f"Measuring {name} (H={H}, V={V}, L={L})...")
        layer_acts = {}
        logits_mems = {}

        for S in SEQ_LENS:
            # Measure layer activation
            try:
                la = measure_layer_act(H, inter, nkv, nh, hd, S)
                layer_acts[S] = la
            except RuntimeError:
                layer_acts[S] = None
                empty_cache()

            # Measure logits
            try:
                lm = measure_logits_mem(H, V, S)
                logits_mems[S] = lm
            except RuntimeError:
                logits_mems[S] = None
                empty_cache()

        all_results[name] = (total_P, H, V, L, layer_acts, logits_mems)

    # Print results
    print()
    print(f"{'='*100}")
    print(f"昇腾 910 ({TOTAL_MEM_GB:.0f} GB) RL 训练策略推荐")
    print(f"{'='*100}")

    for train_gpus, gpu_label in TRAIN_CONFIGS:
        print(f"\n--- {gpu_label}训练 (FSDP={train_gpus}, cpu_offload=True, grad_ckpt=True) ---")
        print(f"{'Model':<18} {'S':>6} {'act':>6} {'lm+lp':>7} {'fsdp':>5} {'peak':>6} {'余量':>5} {'推荐'}")
        print("-" * 75)

        for name, total_P, H, V, L, layer_acts, logits_mems in [
            (n, *all_results[n]) for n, *_ in MODELS
        ]:
            for S in SEQ_LENS:
                la = layer_acts.get(S)
                lm = logits_mems.get(S)

                if la is None or lm is None:
                    # Extrapolate from smaller S if possible
                    # layer_act scales linearly with S
                    smaller = [(s, layer_acts[s]) for s in sorted(layer_acts) if layer_acts[s] is not None and s < S]
                    if len(smaller) >= 1:
                        ref_s, ref_la = smaller[-1]
                        la = ref_la * S / ref_s
                    # logits scales linearly with S
                    smaller_l = [(s, logits_mems[s]) for s in sorted(logits_mems) if logits_mems[s] is not None and s < S]
                    if len(smaller_l) >= 1:
                        ref_s, ref_lm = smaller_l[-1]
                        lm = ref_lm * S / ref_s

                if la is None or lm is None:
                    print(f"{name:<18} {S:>6} {'?':>6} {'?':>7} {'?':>5} {'?':>6} {'?':>5} {'无法测量'}")
                    continue

                peak, total_act, logits_mem, fsdp_oh = estimate_peak(
                    la, lm, L, total_P, H, S, train_gpus
                )
                headroom = TOTAL_MEM_GB - peak

                # Determine recommendation
                if headroom > 10:
                    rec = "FSDP"
                elif headroom > 5:
                    rec = "FSDP*"  # tight
                elif headroom > -5:
                    rec = "SP=2"
                elif headroom > -20:
                    rec = "SP=4"
                else:
                    rec = "SP=8/TP"

                print(f"{name:<18} {S:>6} {total_act:>5.1f}G {logits_mem:>6.1f}G {fsdp_oh:>4.1f}G "
                      f"{peak:>5.0f}G {headroom:>4.0f}G  {rec}")

    print(f"\n{'='*100}")
    print("注: FSDP* = 余量紧张，可能出现 spike; SP=N = 推荐 Sequence Parallel 度数")
    print("    实际悬崖点可能比估算早 ~15% (allocator overhead)")


if __name__ == "__main__":
    main()
