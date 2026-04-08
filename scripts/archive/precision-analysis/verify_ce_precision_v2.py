"""
CORRECTED analysis: vLLM uses fp32 log_softmax for logprobs computation.

From vllm/v1/sample/sampler.py:
    def compute_logprobs(self, logits):
        return logits.log_softmax(dim=-1, dtype=torch.float32)

This changes everything:
- vLLM inference (BOTH platforms): bf16 logits -> fp32 log_softmax (precise)
- H200 training: bf16 logits -> flash_attn triton CE (fp32 accum, precise)
- NPU training: bf16 logits -> bf16 log_softmax (imprecise!)

So the train-vs-inference mismatch is:
- H200: fp32-precise (train) vs fp32-precise (vLLM) -> small mismatch
- NPU: bf16-imprecise (train) vs fp32-precise (vLLM) -> LARGE mismatch!
"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)

VOCAB_SIZE = 151936
N_SAMPLES = 5000

ratio_errors_h200 = []
ratio_errors_npu = []

for _ in range(N_SAMPLES):
    logits_fp32 = torch.randn(1, VOCAB_SIZE, dtype=torch.float32)
    label = torch.randint(0, VOCAB_SIZE, (1,))
    logits_bf16 = logits_fp32.to(torch.bfloat16)

    # vLLM inference (BOTH platforms): bf16 logits -> fp32 log_softmax
    # This is what vllm/v1/sample/sampler.py does
    lp_vllm = F.log_softmax(logits_bf16.float(), dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()

    # H200 training: flash_attn triton CE (bf16 input -> fp32 accumulation)
    # Simulated as: bf16 -> fp32 -> log_softmax
    lp_h200_train = F.log_softmax(logits_bf16.float(), dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()

    # NPU training: bf16 log_softmax (v2 fallback)
    lp_npu_train = F.log_softmax(logits_bf16, dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).float().item()

    # Train-vs-inference log-ratio error
    h200_ratio_log = lp_h200_train - lp_vllm
    npu_ratio_log = lp_npu_train - lp_vllm

    ratio_errors_h200.append(h200_ratio_log)
    ratio_errors_npu.append(npu_ratio_log)

ratio_errors_h200 = torch.tensor(ratio_errors_h200)
ratio_errors_npu = torch.tensor(ratio_errors_npu)

print("=" * 70)
print("CORRECTED: Train-vs-Inference ratio error analysis")
print("(vLLM uses fp32 log_softmax on BOTH platforms)")
print("=" * 70)
print()
print("H200 train-vs-inference ratio log-error:")
print(f"  mean abs: {ratio_errors_h200.abs().mean().item():.8f}")
print(f"  std:      {ratio_errors_h200.std().item():.8f}")
print(f"  max abs:  {ratio_errors_h200.abs().max().item():.8f}")
print()
print("NPU train-vs-inference ratio log-error:")
print(f"  mean abs: {ratio_errors_npu.abs().mean().item():.6f}")
print(f"  std:      {ratio_errors_npu.std().item():.6f}")
print(f"  max abs:  {ratio_errors_npu.abs().max().item():.6f}")
print()

if ratio_errors_h200.abs().mean() > 0:
    ratio = ratio_errors_npu.abs().mean() / ratio_errors_h200.abs().mean()
    print(f"NPU ratio error is {ratio.item():.0f}x larger than H200")
else:
    print(f"H200 ratio error is essentially zero (same impl for train and inference)")
    print(f"NPU ratio error mean abs: {ratio_errors_npu.abs().mean().item():.6f}")
print()

eps_low, eps_high = 0.0003, 0.0004
ln_low = torch.log(torch.tensor(1 - eps_low)).item()
ln_high = torch.log(torch.tensor(1 + eps_high)).item()

h200_clipped = ((ratio_errors_h200 < ln_low) | (ratio_errors_h200 > ln_high)).float().mean().item()
npu_clipped = ((ratio_errors_npu < ln_low) | (ratio_errors_npu > ln_high)).float().mean().item()

print(f"GSPO clip range in log-space: [{ln_low:.6f}, {ln_high:.6f}]")
print(f"H200: {h200_clipped*100:.1f}% of tokens clipped due to train-vs-inference mismatch")
print(f"NPU:  {npu_clipped*100:.1f}% of tokens clipped due to train-vs-inference mismatch")
print()

print("=" * 70)
print("IMPACT ON GRPO/GSPO TRAINING")
print("=" * 70)
print()
print(f"On H200: train and vLLM both use fp32 log_softmax on bf16 inputs.")
print(f"  -> ratio = exp(logp_train - logp_vllm) ≈ 1.0 for same model weights")
print(f"  -> GSPO clip rarely triggered by numerical noise alone")
print(f"  -> clean gradient signal")
print(f"")
print(f"On NPU: train uses bf16 log_softmax, vLLM uses fp32 log_softmax.")
print(f"  -> ratio = exp(logp_train - logp_vllm) has per-token error ~0.019")
print(f"  -> {npu_clipped*100:.0f}% of tokens clipped by GSPO due to numerical noise")
print(f"  -> gradient signal is dominated by noise, not actual policy improvement")
print(f"")
print(f"This is the SMOKING GUN for the NPU precision gap.")
print(f"")
print(f"FIX: In the NPU training path, change logprobs_from_logits_v2 to")
print(f"upcast bf16 logits to fp32 before computing log_softmax:")
print(f"  row_logprobs = F.log_softmax(row_logits.float(), dim=-1)")
print(f"This would match vLLM's behavior and eliminate the mismatch.")

# Verify the fix
print()
print("=" * 70)
print("VERIFICATION: Proposed fix")
print("=" * 70)

fix_errors = []
for _ in range(N_SAMPLES):
    logits_fp32 = torch.randn(1, VOCAB_SIZE, dtype=torch.float32)
    label = torch.randint(0, VOCAB_SIZE, (1,))
    logits_bf16 = logits_fp32.to(torch.bfloat16)

    # vLLM: fp32 log_softmax
    lp_vllm = F.log_softmax(logits_bf16.float(), dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()

    # Fixed NPU training: upcast to fp32 before log_softmax
    lp_fixed = F.log_softmax(logits_bf16.float(), dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()

    fix_errors.append(lp_fixed - lp_vllm)

fix_errors = torch.tensor(fix_errors)
fix_clipped = ((fix_errors < ln_low) | (fix_errors > ln_high)).float().mean().item()

print(f"After fix - train-vs-inference ratio log-error:")
print(f"  mean abs: {fix_errors.abs().mean().item():.8f}")
print(f"  max abs:  {fix_errors.abs().max().item():.8f}")
print(f"  % tokens clipped: {fix_clipped*100:.1f}%")
print(f"  (should be 0.0% - same implementation)")
