#!/usr/bin/env python3
"""
RL 训练显存 dry-run 工具。

模拟一个 RL training step 的完整显存 profile，不需要跑完整训练。
用真实模型、真实序列长度，测量实际 peak 显存，输出是否需要 SP。

Usage:
    # 基本用法 (自动检测设备)
    python rl_mem_dryrun.py --model /path/to/model --max_seq_len 40960

    # 指定训练卡数
    python rl_mem_dryrun.py --model /path/to/model --max_seq_len 40960 --train_gpus 4

    # 测试多个序列长度，找悬崖点
    python rl_mem_dryrun.py --model /path/to/model --max_seq_len 8192,16384,32768,40960
"""
import argparse
import gc
import sys
import time

import torch

# Detect device
if hasattr(torch, "npu") and torch.npu.is_available():
    DEVICE_TYPE = "npu"
    device_mod = torch.npu
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
    device_mod = torch.cuda
else:
    print("No GPU/NPU available. This tool requires a device.")
    sys.exit(1)


def get_mem_gb():
    return device_mod.memory_allocated() / 1e9


def get_peak_gb():
    return device_mod.max_memory_allocated() / 1e9


def reset_peak():
    device_mod.reset_peak_memory_stats()


def empty_cache():
    gc.collect()
    device_mod.empty_cache()


def simulate_rl_step(model, seq_len, device):
    """Simulate one RL training micro-batch: forward (with logits) + backward.

    This mimics what happens in SkyRL/verl's training_step:
    1. model forward → logits (B, S, V)
    2. logprobs_from_logits(logits, labels) → log_probs (B, S)
    3. entropy from logits
    4. compute loss from log_probs
    5. loss.backward()
    """
    B = 1  # micro_batch_size = 1 in RL training
    V = model.config.vocab_size

    input_ids = torch.randint(0, V, (B, seq_len), device=device)
    attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
    labels = torch.roll(input_ids, -1, dims=1)

    # === Forward: model produces logits ===
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits  # (B, S, V) — the memory killer

    # === logprobs_from_logits (simplified, matches SkyRL _v2 behavior) ===
    # Per-row loop with fp32 upcast
    log_probs_list = []
    for row_logits, row_labels in zip(logits, labels):
        row_lp = torch.nn.functional.log_softmax(row_logits.float(), dim=-1)
        row_lp_gathered = row_lp.gather(-1, row_labels.unsqueeze(-1)).squeeze(-1)
        log_probs_list.append(row_lp_gathered)
    log_probs = torch.stack(log_probs_list)

    # === Entropy (simplified chunked version) ===
    chunk_size = 1024
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_lp = torch.nn.functional.log_softmax(logits[:, start:end], dim=-1)
        chunk_ent = -(chunk_lp.exp() * chunk_lp).sum(-1)
        del chunk_lp, chunk_ent

    # === Compute loss (simplified PPO/GSPO) ===
    # In real training: ratio = exp(log_probs - old_log_probs), then clipped loss
    # Here we just use log_probs.sum() as a proxy that creates the same autograd graph
    loss = log_probs.sum()

    # Record peak after forward (before backward)
    device_mod.synchronize()
    peak_after_fwd = get_peak_gb()

    # === Backward ===
    loss.backward()
    device_mod.synchronize()
    peak_after_bwd = get_peak_gb()

    # Cleanup
    del output, logits, log_probs, loss, input_ids, attention_mask, labels
    model.zero_grad()
    empty_cache()

    return peak_after_fwd, peak_after_bwd


def run_dryrun(model_path, seq_lens, train_gpus, device_id=0):
    device = f"{DEVICE_TYPE}:{device_id}"
    device_mod.set_device(device_id)

    total_mem = device_mod.get_device_properties(device_id).total_memory / 1e9

    print(f"Device: {DEVICE_TYPE} ({total_mem:.0f} GB)")
    print(f"Model: {model_path}")
    print(f"Train GPUs: {train_gpus}")
    print()

    # Load model
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"Config: hidden={config.hidden_size}, vocab={config.vocab_size}, "
          f"layers={config.num_hidden_layers}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    empty_cache()
    reset_peak()
    model_mem = get_mem_gb()
    print(f"Model memory: {model_mem:.1f} GB")
    print()

    # Note: this is single-GPU simulation. In FSDP training:
    # - model params are sharded (÷ train_gpus), but all-gathered per layer during forward
    # - optimizer states are sharded or offloaded
    # - logits and activations are NOT sharded (unless SP is used)
    # So single-GPU peak for logits+activations is representative of FSDP training.
    # We add an estimated FSDP overhead.
    fsdp_overhead_gb = config.num_hidden_layers * (
        config.hidden_size * config.hidden_size * 4 * 2  # rough single-layer param size
    ) / config.num_hidden_layers * 2 / 1e9  # all-gather buffer
    fsdp_overhead_gb = max(fsdp_overhead_gb, 2.0)  # at least 2 GB
    fsdp_overhead_gb += (  # sharded gradients
        sum(p.numel() for p in model.parameters()) * 2 / train_gpus / 1e9
    )

    results = []
    print(f"{'S':>6}  {'fwd_peak':>9}  {'bwd_peak':>9}  {'+ FSDP':>8}  {'余量':>6}  {'状态'}")
    print("-" * 60)

    for S in seq_lens:
        empty_cache()
        reset_peak()

        try:
            fwd_peak, bwd_peak = simulate_rl_step(model, S, device)
            # Subtract model_mem to get the "variable" part, then add FSDP overhead
            # In FSDP, model params are sharded, so subtract full model and add shard
            fsdp_peak = bwd_peak - model_mem + model_mem / train_gpus + fsdp_overhead_gb
            headroom = total_mem - fsdp_peak
            if headroom > 10:
                status = "✓ 安全"
            elif headroom > 5:
                status = "⚠ 紧张 (可能 spike)"
            elif headroom > 0:
                status = "✗ 危险 (会 spike)"
            else:
                status = "✗ OOM"
            results.append((S, fwd_peak, bwd_peak, fsdp_peak, headroom, status))
            print(f"{S:>6}  {fwd_peak:>8.1f}G  {bwd_peak:>8.1f}G  {fsdp_peak:>7.1f}G  {headroom:>5.1f}G  {status}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results.append((S, None, None, None, None, "OOM"))
                print(f"{S:>6}  {'OOM':>9}  {'OOM':>9}  {'OOM':>8}  {'<0':>6}  ✗ OOM")
                empty_cache()
            else:
                raise

    # Find cliff point
    print()
    cliff = None
    for S, fwd, bwd, fsdp, headroom, status in results:
        if headroom is not None and headroom < 5:
            cliff = S
            break

    if cliff:
        print(f"⚠ 悬崖点: S ≈ {cliff}")
    else:
        # Check if any OOM happened
        oom_at = None
        for S, fwd, bwd, fsdp, headroom, status in results:
            if status == "OOM":
                oom_at = S
                break
        if oom_at:
            # OOM on single GPU = definitely needs SP in FSDP training too
            # Estimate cliff as the last successful S
            last_ok = None
            for S, fwd, bwd, fsdp, headroom, status in results:
                if headroom is not None and headroom > 0:
                    last_ok = S
            cliff = last_ok or seq_lens[0]
            print(f"⚠ 悬崖点: S ≈ {cliff} ~ {oom_at}")
            print(f"  单卡在 S={oom_at} 时 OOM (FSDP 训练中可能在更长序列才 spike)")
        else:
            print("✓ 所有测试序列长度都安全，不需要 SP")

    if cliff:
        max_S = max(s for s, *_ in results)
        print(f"  序列长度超过 {cliff} 时训练会出现 spike 或 OOM")
        print()
        print("推荐方案:")
        for sp in [2, 4, 8]:
            if sp > train_gpus:
                break
            max_supported = cliff * sp
            if max_S <= max_supported:
                dp = train_gpus // sp
                print(f"  → SP={sp} (DP={dp}): 可支持到 S≈{max_supported}")
                break
        else:
            print(f"  SP={train_gpus} 可能仍不够，考虑增加训练卡数或用 TP+SP")

    # Cleanup
    del model
    empty_cache()


def main():
    parser = argparse.ArgumentParser(description="RL 训练显存 dry-run")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--max_seq_len", required=True,
                        help="序列长度，逗号分隔多个值。如: 8192,16384,32768,40960")
    parser.add_argument("--train_gpus", type=int, default=4, help="训练卡数 (默认 4)")
    parser.add_argument("--device_id", type=int, default=0, help="测试用的设备 ID")
    args = parser.parse_args()

    seq_lens = [int(s.strip()) for s in args.max_seq_len.split(",")]
    run_dryrun(args.model, seq_lens, args.train_gpus, args.device_id)


if __name__ == "__main__":
    main()
