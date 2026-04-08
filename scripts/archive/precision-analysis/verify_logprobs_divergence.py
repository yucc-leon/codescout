"""
Verify numerical divergence between flash_attn cross_entropy and v2 fallback
for logprobs computation.

This runs on CPU to simulate the difference between:
- H200 path: flash_attn triton cross_entropy (fused kernel)
- NPU path: F.log_softmax + gather (v2 fallback, bf16 branch)

We can't run the actual triton kernel on CPU, but we CAN compare:
1. fp32 log_softmax (reference)
2. bf16 log_softmax (what NPU actually does)
3. bf16 logsumexp (what would happen if bf16 took the fp32 branch)

This quantifies the precision loss from bf16 log_softmax vs fp32 reference,
which is the core of the NPU vs H200 divergence.
"""

import torch
import torch.nn.functional as F
import sys

torch.manual_seed(42)

# Simulate Qwen3-4B vocab size and typical sequence
VOCAB_SIZE = 151936  # Qwen3 vocab
SEQ_LENS = [1, 16, 128, 1024]
BATCH = 1

print("=" * 70)
print("Logprobs computation path divergence analysis")
print("=" * 70)

for seq_len in SEQ_LENS:
    # Generate random logits in fp32 (ground truth)
    logits_fp32 = torch.randn(BATCH, seq_len, VOCAB_SIZE, dtype=torch.float32)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH, seq_len))

    # === Path 1: fp32 reference (closest to flash_attn triton kernel behavior) ===
    # The triton kernel does computation in fp32 internally
    logprobs_fp32 = F.log_softmax(logits_fp32, dim=-1)
    selected_fp32 = logprobs_fp32.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # === Path 2: bf16 log_softmax (what NPU v2 fallback does) ===
    logits_bf16 = logits_fp32.to(torch.bfloat16)
    logprobs_bf16_list = []
    for row_logits, row_labels in zip(logits_bf16.view(-1, VOCAB_SIZE), labels.view(-1)):
        row_logprobs = F.log_softmax(row_logits.unsqueeze(0), dim=-1)
        row_lp = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        logprobs_bf16_list.append(row_lp)
    selected_bf16 = torch.cat(logprobs_bf16_list).view(BATCH, seq_len).float()

    # === Path 3: bf16 logits -> fp32 log_softmax (hypothetical better path) ===
    logits_bf16_to_fp32 = logits_bf16.float()
    logprobs_upcast = F.log_softmax(logits_bf16_to_fp32, dim=-1)
    selected_upcast = logprobs_upcast.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # === Compare ===
    diff_bf16_vs_fp32 = (selected_bf16 - selected_fp32).abs()
    diff_upcast_vs_fp32 = (selected_upcast - selected_fp32).abs()

    print(f"\nSeq len = {seq_len}, vocab = {VOCAB_SIZE}")
    print(f"  bf16 log_softmax vs fp32 reference (NPU v2 path error):")
    print(f"    max abs diff:  {diff_bf16_vs_fp32.max().item():.6f}")
    print(f"    mean abs diff: {diff_bf16_vs_fp32.mean().item():.6f}")
    print(f"    std abs diff:  {diff_bf16_vs_fp32.std().item():.6f}")

    print(f"  bf16->fp32 upcast log_softmax vs fp32 reference:")
    print(f"    max abs diff:  {diff_upcast_vs_fp32.max().item():.6f}")
    print(f"    mean abs diff: {diff_upcast_vs_fp32.mean().item():.6f}")
    print(f"    std abs diff:  {diff_upcast_vs_fp32.std().item():.6f}")

    # Relative error on the logprobs themselves
    rel_err = (diff_bf16_vs_fp32 / selected_fp32.abs().clamp(min=1e-8))
    print(f"  Relative error (bf16 vs fp32):")
    print(f"    max:  {rel_err.max().item():.6f}")
    print(f"    mean: {rel_err.mean().item():.6f}")

    # Show how this affects policy gradient
    # In GRPO, the gradient is proportional to advantage * (logprob_new - logprob_old)
    # If logprob computation has error e, the ratio exp(logprob_new - logprob_old)
    # gets multiplied by exp(e), which for small e ≈ 1 + e
    print(f"  Impact on policy ratio exp(logp_new - logp_old):")
    print(f"    max ratio perturbation: exp({diff_bf16_vs_fp32.max().item():.4f}) = {torch.exp(diff_bf16_vs_fp32.max()).item():.6f}")
    print(f"    (1.0 = no perturbation)")

print("\n" + "=" * 70)
print("Key insight: The bf16 log_softmax path (NPU) introduces per-token")
print("logprob errors. Over thousands of tokens per rollout and hundreds")
print("of training steps, these errors compound in the policy gradient,")
print("leading to different optimization trajectories.")
print()
print("The flash_attn triton cross_entropy kernel (H200) computes the")
print("cross entropy in a single fused pass with better numerical stability")
print("(it uses online softmax with fp32 accumulation internally).")
print("=" * 70)
