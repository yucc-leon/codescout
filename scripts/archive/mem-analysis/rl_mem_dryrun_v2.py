#!/usr/bin/env python3
"""
RL 训练显存 dry-run v2 — 精确版。

思路：不加载完整模型（会 OOM），而是分别测量各组件的实际显存：
1. 单层 transformer forward+backward → 得到 per-layer activation 显存
2. lm_head matmul + logprobs 计算 → 得到 logits 显存
3. 用模型 config 计算 FSDP 参数/梯度/optimizer 的理论值（这些是精确的）
4. 组合得到 FSDP 训练的 peak 显存

这样避免了单卡放完整模型的限制，估算更接近真实 FSDP 训练。

Usage:
    python rl_mem_dryrun_v2.py --model /path/to/model --max_seq_len 40960 --train_gpus 4
"""
import argparse
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


def empty_cache():
    gc.collect()
    dev.empty_cache()


def measure_layer_activation(model_path, seq_lens, device_id):
    """Measure per-layer activation memory at different sequence lengths."""
    from transformers import AutoModelForCausalLM
    dev.set_device(device_id)
    device = f"{DEVICE_TYPE}:{device_id}"

    # Load full model, extract one layer, delete the rest
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                  trust_remote_code=True).to(device)
    layer = model.model.layers[0]
    embed = model.model.embed_tokens
    rotary = model.model.rotary_emb
    embed.requires_grad_(False)
    rotary.requires_grad_(False)

    # Delete all other layers to free memory
    del model.model.layers[1:]
    del model
    empty_cache()

    results = {}
    for S in seq_lens:
        empty_cache()
        dev.reset_peak_memory_stats()

        try:
            input_ids = torch.randint(0, 1000, (1, S), device=device)
            pos_ids = torch.arange(S, device=device).unsqueeze(0)
            with torch.no_grad():
                hidden = embed(input_ids)
                cos, sin = rotary(hidden, pos_ids)
            pos_emb = (cos.detach(), sin.detach())
            hidden = hidden.detach().requires_grad_(True)
            del input_ids

            mem_before = dev.memory_allocated() / 1e9

            # Forward + backward
            out = layer(hidden, position_ids=pos_ids, position_embeddings=pos_emb)
            out_t = out[0] if isinstance(out, tuple) else out
            out_t.sum().backward()
            dev.synchronize()

            peak = dev.max_memory_allocated() / 1e9
            act_mem = peak - mem_before
            results[S] = act_mem

            del hidden, pos_emb, pos_ids, out, out_t
            layer.zero_grad()
            empty_cache()
        except RuntimeError:
            results[S] = None
            empty_cache()

    del layer, embed, rotary
    empty_cache()
    return results


def measure_logits_mem(vocab_size, hidden_size, seq_lens, device_id):
    """Measure actual logits + logprobs computation memory."""
    dev.set_device(device_id)
    device = f"{DEVICE_TYPE}:{device_id}"

    # Create a standalone lm_head
    lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False,
                               dtype=torch.bfloat16, device=device)
    lm_head_mem = dev.memory_allocated() / 1e9

    results = {}
    for S in seq_lens:
        empty_cache()
        dev.reset_peak_memory_stats()

        try:
            hidden = torch.randn(1, S, hidden_size, dtype=torch.bfloat16, device=device)
            labels = torch.randint(0, vocab_size, (1, S), device=device)

            mem_before = dev.memory_allocated() / 1e9

            # Simulate logits + logprobs (matches SkyRL _v2 behavior)
            logits = lm_head(hidden)
            # Per-row fp32 log_softmax + gather (like logprobs_from_logits_v2)
            lp_list = []
            for row_l, row_lab in zip(logits, labels):
                row_lp = torch.nn.functional.log_softmax(row_l.float(), dim=-1)
                lp_list.append(row_lp.gather(-1, row_lab.unsqueeze(-1)).squeeze(-1))
            log_probs = torch.stack(lp_list)
            # Entropy (chunked, like SkyRL)
            for start in range(0, S, 1024):
                end = min(start + 1024, S)
                clp = torch.nn.functional.log_softmax(logits[:, start:end], dim=-1)
                _ = -(clp.exp() * clp).sum(-1)
                del clp

            dev.synchronize()
            peak = dev.max_memory_allocated() / 1e9
            logits_mem = peak - mem_before

            # Also measure with backward (logits stays alive for autograd)
            loss = log_probs.sum()
            loss.backward()
            dev.synchronize()
            peak_bwd = dev.max_memory_allocated() / 1e9
            logits_mem_bwd = peak_bwd - mem_before

            results[S] = (logits_mem, logits_mem_bwd)

            del hidden, labels, logits, log_probs, loss
            lm_head.zero_grad()
            empty_cache()
        except RuntimeError:
            results[S] = None
            empty_cache()

    del lm_head
    empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="RL 训练显存 dry-run v2 (精确版)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_seq_len", required=True,
                        help="逗号分隔，如 8192,16384,32768,40960")
    parser.add_argument("--train_gpus", type=int, default=4)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--cpu_offload", action="store_true", default=True,
                        help="是否 cpu_offload optimizer (默认 True)")
    args = parser.parse_args()

    seq_lens = sorted(int(s.strip()) for s in args.max_seq_len.split(","))
    device_id = args.device_id
    dev.set_device(device_id)
    total_mem_gb = dev.get_device_properties(device_id).total_memory / 1e9

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    H = config.hidden_size
    V = config.vocab_size
    L = config.num_hidden_layers
    total_params = sum(1 for _ in range(1))  # placeholder, compute below

    # Count params from config
    # Rough: embedding + L * (attn + mlp) + lm_head
    from transformers import AutoModelForCausalLM
    tmp = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                trust_remote_code=True)
    total_params = sum(p.numel() for p in tmp.parameters())
    del tmp
    empty_cache()

    P = total_params
    fsdp = args.train_gpus

    print(f"{'='*65}")
    print(f"RL Memory Dry-Run v2")
    print(f"  Model: {args.model}")
    print(f"  Config: H={H}, V={V}, L={L}, params={P/1e9:.1f}B")
    print(f"  Device: {DEVICE_TYPE} ({total_mem_gb:.0f} GB)")
    print(f"  Train GPUs: {fsdp}, cpu_offload={args.cpu_offload}")
    print(f"{'='*65}")

    # === Step 1: Measure per-layer activation ===
    print("\n[1/2] Measuring per-layer activations...")
    layer_results = measure_layer_activation(args.model, seq_lens, device_id)

    # === Step 2: Measure logits + logprobs ===
    print("[2/2] Measuring logits + logprobs...")
    logits_results = measure_logits_mem(V, H, seq_lens, device_id)

    # === Step 3: Compute FSDP overhead (theoretical, precise) ===
    # These are exact because they only depend on param count and FSDP config
    param_shard_gb = P * 2 / fsdp / 1e9          # bf16 sharded params
    grad_shard_gb = P * 2 / fsdp / 1e9            # bf16 sharded gradients
    fsdp_allgather_gb = (P / L) * 2 / 1e9         # single layer all-gather (bf16)
    if args.cpu_offload:
        opt_gb = 0  # optimizer on CPU
    else:
        opt_gb = P * 4 * 2 / fsdp / 1e9           # Adam fp32, 2 states, sharded

    fsdp_fixed = param_shard_gb + grad_shard_gb + fsdp_allgather_gb + opt_gb + 1.0  # +1 GB buffer

    print(f"\nFSDP fixed overhead: {fsdp_fixed:.1f} GB")
    print(f"  param shard: {param_shard_gb:.1f}, grad shard: {grad_shard_gb:.1f}, "
          f"all-gather buf: {fsdp_allgather_gb:.2f}, optimizer: {opt_gb:.1f}")

    # === Step 4: Combine ===
    print(f"\n{'S':>6}  {'layer_act':>10}  {'L*act':>8}  {'logits':>8}  {'FSDP':>6}  {'peak':>6}  {'余量':>6}  {'状态'}")
    print("-" * 75)

    cliff = None
    for S in seq_lens:
        layer_act = layer_results.get(S)
        logits_data = logits_results.get(S)

        if layer_act is None or logits_data is None:
            print(f"{S:>6}  {'OOM':>10}  {'':>8}  {'OOM':>8}  {'':>6}  {'OOM':>6}  {'<0':>6}  ✗ OOM")
            if cliff is None:
                # Find last OK
                for prev_S in seq_lens:
                    if prev_S >= S:
                        break
                    if layer_results.get(prev_S) is not None:
                        cliff = prev_S
            continue

        # In FSDP training with grad ckpt:
        # - Forward: saves each layer's input hidden state = L * S * H * 2 bytes
        # - Backward: recomputes one layer at a time = layer_act (measured)
        # - Peak is during backward: all saved inputs + one layer recompute + logits
        ckpt_saved = L * S * H * 2 / 1e9  # all layers' saved hidden states
        ckpt_saved = L * S * H * 2 / 1e9  # all layers' saved hidden states
        total_act = ckpt_saved + layer_act  # saved + one layer recompute
        _, logits_bwd = logits_data  # logits mem including backward

        peak = total_act + logits_bwd + fsdp_fixed
        headroom = total_mem_gb - peak

        if headroom > 10:
            status = "✓ 安全"
        elif headroom > 5:
            status = "⚠ 紧张"
        elif headroom > 0:
            status = "✗ 危险 (spike)"
            if cliff is None:
                cliff = S
        else:
            status = "✗ OOM"
            if cliff is None:
                # last OK
                for prev_S in seq_lens:
                    if prev_S >= S:
                        break
                    la = layer_results.get(prev_S)
                    ld = logits_results.get(prev_S)
                    if la is not None and ld is not None:
                        cliff = prev_S

        print(f"{S:>6}  {layer_act:>9.2f}G  {total_act:>7.1f}G  {logits_bwd:>7.1f}G  "
              f"{fsdp_fixed:>5.1f}G  {peak:>5.1f}G  {headroom:>5.1f}G  {status}")

    # === Recommendation ===
    print()
    if cliff is None:
        print("✓ 所有序列长度都安全，不需要 SP")
    else:
        max_S = max(seq_lens)
        print(f"⚠ 悬崖点: S ≈ {cliff}")
        print()
        print("推荐:")
        for sp in [2, 4, 8]:
            if sp > fsdp:
                break
            # With SP=k, each GPU sees S/k tokens
            # Check if the cliff * k covers our max_seq_len
            if cliff * sp >= max_S:
                dp = fsdp // sp
                print(f"  → SP={sp} (DP={dp})")
                print(f"    每卡处理 S/{sp} tokens, 可支持到 S≈{cliff * sp}")
                break
        else:
            print(f"  SP={fsdp} 可能不够，需要更多卡或 TP+SP")


if __name__ == "__main__":
    main()
