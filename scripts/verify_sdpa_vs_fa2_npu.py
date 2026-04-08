#!/usr/bin/env python3
"""
Compare logprobs computed via SDPA vs FA2 (varlen) attention paths on NPU.

This tests whether the attention implementation change (sdpa → flash_attention_2)
introduced by SP=2 causes systematic logprob differences that could explain
the reward gap between config A and config B.

On NPU, both paths ultimately call npu_fusion_attention, but with different:
  - Input layouts: BNSD (sdpa) vs TND (fa2/varlen)
  - Masking: attention_mask tensor (sdpa) vs cu_seqlens (fa2/varlen)

Usage:
  # Single GPU test (no distributed needed)
  python scripts/verify_sdpa_vs_fa2_npu.py \
    --model /sharedata/liyuchen/models/Qwen3-4B-Instruct-2507 \
    --seq_lens 1024,4096,8192,16384
"""

import argparse
import sys
import os

# NPU patch must be first
import npu_support.patch_cuda  # noqa: F401, isort:skip

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def create_test_input(tokenizer, seq_len, device):
    """Create a deterministic test input of given length."""
    # Use a fixed prompt and pad/truncate to desired length
    text = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n" * 200
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < seq_len:
        tokens = tokens * (seq_len // len(tokens) + 1)
    tokens = tokens[:seq_len]
    input_ids = torch.tensor([tokens], device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def compute_logprobs_sdpa(model, input_ids, attention_mask):
    """Compute logprobs using SDPA attention path."""
    model.config._attn_implementation = "sdpa"
    # Force all attention layers to use sdpa
    for module in model.modules():
        if hasattr(module, 'config') and hasattr(module.config, '_attn_implementation'):
            module.config._attn_implementation = "sdpa"

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    logits = output.logits  # (1, S, V)

    # Compute logprobs for next-token prediction
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs, logits


def compute_logprobs_fa2(model, input_ids, attention_mask):
    """Compute logprobs using flash_attention_2 path (varlen on NPU)."""
    model.config._attn_implementation = "flash_attention_2"
    for module in model.modules():
        if hasattr(module, 'config') and hasattr(module.config, '_attn_implementation'):
            module.config._attn_implementation = "flash_attention_2"

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    with torch.no_grad():
        # FA2 path: don't pass attention_mask, use position_ids only
        output = model(input_ids, attention_mask=None, position_ids=position_ids)
    logits = output.logits

    labels = torch.roll(input_ids, shifts=-1, dims=1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs, logits


def compute_logprobs_fa2_with_mask(model, input_ids, attention_mask):
    """Compute logprobs using flash_attention_2 path WITH attention_mask."""
    model.config._attn_implementation = "flash_attention_2"
    for module in model.modules():
        if hasattr(module, 'config') and hasattr(module.config, '_attn_implementation'):
            module.config._attn_implementation = "flash_attention_2"

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    logits = output.logits

    labels = torch.roll(input_ids, shifts=-1, dims=1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seq_lens", type=str, default="512,1024,4096")
    parser.add_argument("--device", type=str, default="npu:0")
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    device = args.device

    print(f"Loading model: {args.model}")
    print(f"Device: {device}")
    print(f"Sequence lengths: {seq_lens}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load model in bf16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # start with sdpa
    ).to(device).eval()

    print(f"Model loaded. Config attn_implementation: {model.config._attn_implementation}")
    print()

    for seq_len in seq_lens:
        print(f"{'='*60}")
        print(f"Sequence length: {seq_len}")
        print(f"{'='*60}")

        input_ids, attention_mask = create_test_input(tokenizer, seq_len, device)

        # Path 1: SDPA
        logprobs_sdpa, logits_sdpa = compute_logprobs_sdpa(model, input_ids, attention_mask)

        # Path 2: FA2 (no attention_mask, position_ids only — like sample_packing)
        logprobs_fa2, logits_fa2 = compute_logprobs_fa2(model, input_ids, attention_mask)

        # Path 3: FA2 with attention_mask (like non-packing FA2)
        logprobs_fa2_mask, logits_fa2_mask = compute_logprobs_fa2_with_mask(model, input_ids, attention_mask)

        # Compare logits (before softmax)
        logit_diff_fa2 = (logits_sdpa.float() - logits_fa2.float()).abs()
        logit_diff_fa2_mask = (logits_sdpa.float() - logits_fa2_mask.float()).abs()

        # Compare logprobs
        lp_diff_fa2 = (logprobs_sdpa - logprobs_fa2).abs()
        lp_diff_fa2_mask = (logprobs_sdpa - logprobs_fa2_mask).abs()

        # Exclude last position (roll artifact)
        lp_diff_fa2 = lp_diff_fa2[:, :-1]
        lp_diff_fa2_mask = lp_diff_fa2_mask[:, :-1]

        print(f"\n  SDPA vs FA2 (no mask, like sample_packing path):")
        print(f"    Logits  - max: {logit_diff_fa2.max():.6f}, mean: {logit_diff_fa2.mean():.6f}")
        print(f"    Logprobs - max: {lp_diff_fa2.max():.6f}, mean: {lp_diff_fa2.mean():.6f}")
        print(f"    Logprobs > 0.01: {(lp_diff_fa2 > 0.01).sum().item()} / {lp_diff_fa2.numel()}")
        print(f"    Logprobs > 0.001: {(lp_diff_fa2 > 0.001).sum().item()} / {lp_diff_fa2.numel()}")

        print(f"\n  SDPA vs FA2 (with mask):")
        print(f"    Logits  - max: {logit_diff_fa2_mask.max():.6f}, mean: {logit_diff_fa2_mask.mean():.6f}")
        print(f"    Logprobs - max: {lp_diff_fa2_mask.max():.6f}, mean: {lp_diff_fa2_mask.mean():.6f}")
        print(f"    Logprobs > 0.01: {(lp_diff_fa2_mask > 0.01).sum().item()} / {lp_diff_fa2_mask.numel()}")
        print(f"    Logprobs > 0.001: {(lp_diff_fa2_mask > 0.001).sum().item()} / {lp_diff_fa2_mask.numel()}")

        # GSPO impact analysis
        # GSPO clip range: [1-0.0003, 1+0.0004] → log range: [-0.0003, 0.0004]
        # If logprob diff > 0.0003, it can cause spurious clipping
        gspo_threshold = 0.0003
        n_above = (lp_diff_fa2 > gspo_threshold).sum().item()
        pct_above = n_above / lp_diff_fa2.numel() * 100
        print(f"\n  GSPO impact (threshold={gspo_threshold}):")
        print(f"    Tokens with |diff| > {gspo_threshold}: {n_above} / {lp_diff_fa2.numel()} ({pct_above:.1f}%)")
        if pct_above > 1:
            print(f"    ⚠️  SIGNIFICANT: {pct_above:.1f}% of tokens have logprob diff exceeding GSPO clip range")
            print(f"    This means the attention implementation change alone can cause")
            print(f"    different clipping behavior, leading to different gradient signals.")
        else:
            print(f"    ✓ Negligible: only {pct_above:.1f}% of tokens affected")

        print()

        # Clean up
        del logprobs_sdpa, logprobs_fa2, logprobs_fa2_mask
        del logits_sdpa, logits_fa2, logits_fa2_mask
        torch.npu.empty_cache() if hasattr(torch, 'npu') else torch.cuda.empty_cache()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("If SDPA vs FA2 logprob differences are significant (>0.001 mean),")
    print("the attention implementation change is the root cause of the")
    print("reward gap between config A (SDPA) and config B (FA2).")
    print()
    print("Fix: Decouple SP from flash_attention_2. Allow SP=2 with SDPA.")
    print("This requires modifying SkyRL's model_wrapper.py to support")
    print("sample_packing with SDPA (using block-diagonal attention mask).")
    print()
    print("Alternative fix: Keep FA2 but add a logprob correction term")
    print("to compensate for the systematic bias.")


if __name__ == "__main__":
    main()
