"""
Corrected GRPO impact analysis.

Key insight: In async GRPO, logp_old is computed by the INFERENCE engine (vLLM)
and logp_new is computed by the TRAINING engine (HF model wrapper).

On H200:
  - logp_old: vLLM with CUDA kernels (fp16/bf16 inference)
  - logp_new: flash_attn triton cross_entropy (fused, fp32 accumulation)

On NPU:
  - logp_old: vLLM-Ascend with NPU kernels (bf16 inference)
  - logp_new: F.log_softmax in bf16 (v2 fallback)

The ratio = exp(logp_new - logp_old) is computed from TWO DIFFERENT engines,
so errors don't cancel out. But the question is whether the H200 path also
has a similar inference-vs-training mismatch.

Actually, the more important question is: does the bf16 logprobs error
affect the GRADIENT DIRECTION, not just the ratio magnitude?

In GSPO (the actual loss type used), the loss is:
  L = -min(ratio * A, clip(ratio) * A)
where ratio = exp(sum_t (logp_new_t - logp_old_t))

The clip range is [1 - eps_low, 1 + eps_high] = [0.9997, 1.0004].

But wait - this is per-SEQUENCE ratio, not per-token. Let me re-check
how the loss is actually computed in the codebase.
"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)

VOCAB_SIZE = 151936

# First, let's understand the actual error characteristics better
print("=" * 70)
print("Part 1: Per-token logprob error characteristics")
print("=" * 70)

N_SAMPLES = 5000
errors_signed = []
for _ in range(N_SAMPLES):
    logits_fp32 = torch.randn(1, VOCAB_SIZE, dtype=torch.float32)
    label = torch.randint(0, VOCAB_SIZE, (1,))

    lp_fp32 = F.log_softmax(logits_fp32, dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1)
    logits_bf16 = logits_fp32.to(torch.bfloat16)
    lp_bf16 = F.log_softmax(logits_bf16, dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).float()

    errors_signed.append((lp_bf16 - lp_fp32).item())

errors_signed = torch.tensor(errors_signed)
print(f"Signed error stats (bf16 - fp32):")
print(f"  mean:   {errors_signed.mean().item():.6f}")
print(f"  std:    {errors_signed.std().item():.6f}")
print(f"  median: {errors_signed.median().item():.6f}")
print(f"  min:    {errors_signed.min().item():.6f}")
print(f"  max:    {errors_signed.max().item():.6f}")
print(f"  % positive: {(errors_signed > 0).float().mean().item()*100:.1f}%")
print(f"  % negative: {(errors_signed < 0).float().mean().item()*100:.1f}%")

print()
print("=" * 70)
print("Part 2: Impact on per-token ratio (the actual gradient-relevant quantity)")
print("=" * 70)
print()
print("In the actual training code, the ratio is computed per-token, not per-sequence.")
print("The GSPO loss operates on per-token log-ratios: logp_new_t - logp_old_t")
print()

# The real question: how does the bf16 error in logp_new affect the per-token ratio?
# ratio_t = exp(logp_new_t - logp_old_t)
# With bf16 error: ratio_t_noisy = exp((logp_new_t + e_t) - logp_old_t) = ratio_t * exp(e_t)
# For e_t ~ N(0.0007, 0.023), exp(e_t) ~ 1 + e_t for small e_t

print("Per-token ratio perturbation from bf16 error:")
for percentile in [50, 90, 95, 99]:
    e = errors_signed.abs().quantile(percentile / 100).item()
    print(f"  {percentile}th percentile |e|: {e:.4f} -> ratio multiplied by [{torch.exp(torch.tensor(-e)).item():.4f}, {torch.exp(torch.tensor(e)).item():.4f}]")

print()
print("=" * 70)
print("Part 3: Does this matter for GSPO with eps_clip=[0.0003, 0.0004]?")
print("=" * 70)
print()

# The clip range is [1-0.0003, 1+0.0004] = [0.9997, 1.0004]
# This is EXTREMELY tight. Even a tiny logprob error can push the ratio outside.
eps_low, eps_high = 0.0003, 0.0004

# What fraction of per-token ratios would be clipped due to bf16 error alone?
# If the true ratio is exactly 1.0 (logp_new == logp_old), then:
# noisy_ratio = exp(e_t)
# clipped if exp(e_t) < 0.9997 or exp(e_t) > 1.0004
# i.e., e_t < ln(0.9997) = -0.0003 or e_t > ln(1.0004) = 0.0004

ln_low = torch.log(torch.tensor(1 - eps_low)).item()
ln_high = torch.log(torch.tensor(1 + eps_high)).item()

frac_clipped = ((errors_signed < ln_low) | (errors_signed > ln_high)).float().mean().item()
print(f"Clip thresholds in logprob space: [{ln_low:.6f}, {ln_high:.6f}]")
print(f"Fraction of tokens where bf16 error ALONE would cause clipping: {frac_clipped*100:.1f}%")
print()

frac_in_range = ((errors_signed >= ln_low) & (errors_signed <= ln_high)).float().mean().item()
print(f"Fraction of tokens where bf16 error is within clip range: {frac_in_range*100:.1f}%")
print()

print("CONCLUSION:")
if frac_clipped > 0.5:
    print(f"  {frac_clipped*100:.0f}% of tokens have bf16 logprob errors larger than the")
    print(f"  GSPO clip range [{1-eps_low}, {1+eps_high}].")
    print(f"  This means the bf16 error is MUCH LARGER than the signal the")
    print(f"  optimizer is trying to learn from.")
    print(f"  ")
    print(f"  However, this affects BOTH H200 and NPU equally IF both use")
    print(f"  the same logprobs function. The key difference is:")
    print(f"  - H200: flash_attn triton CE (fp32 accumulation) -> less error")
    print(f"  - NPU: bf16 log_softmax -> more error")
    print(f"  ")
    print(f"  The DIFFERENTIAL error between the two paths is what matters.")
else:
    print(f"  Only {frac_clipped*100:.0f}% of tokens affected. The error is small")
    print(f"  relative to the clip range.")

print()
print("=" * 70)
print("Part 4: What about the inference engine (vLLM) logprobs?")
print("=" * 70)
print()
print("Both H200 and NPU use vLLM for inference, which computes logprobs")
print("in its own way (likely bf16 on both platforms).")
print("The ratio = exp(logp_train - logp_vllm) has error from BOTH sides.")
print()
print("On H200: logp_train uses flash_attn CE (more precise)")
print("On NPU:  logp_train uses bf16 log_softmax (less precise)")
print()
print("So the NPU ratio has MORE noise than H200 ratio.")
print("With the extremely tight GSPO clip range, this extra noise means:")
print("  - More tokens get clipped on NPU -> weaker gradient signal")
print("  - Gradient direction is noisier -> slower/worse convergence")
print("  - This explains: lower rewards, faster entropy collapse,")
print("    more grad_norm spikes")
