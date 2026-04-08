"""
CPU simulation of the precision analysis.

We can't run on NPU (occupied), but we can reason about the behavior:

torch_npu.npu_cross_entropy_loss documentation says:
- Input dtype: FLOAT16, FLOAT32, BFLOAT16
- Output dtype: same as input
- It's a fused log_softmax + nll_loss

The key question: does it do fp32 accumulation internally?

From the doc: "输出数据类型与input相同" (output dtype same as input)
This suggests it does NOT upcast internally - if input is bf16, output is bf16.

Let's simulate both scenarios:
A) npu_ce does fp32 accumulation (like flash_attn triton CE) -> small error
B) npu_ce does bf16 computation (like bf16 log_softmax) -> large error

And compare with vLLM's fp32 logprobs to determine the train-vs-inference mismatch.
"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)

VOCAB_SIZE = 151936
N_SAMPLES = 5000

# Scenario A: npu_ce with fp32 accumulation (optimistic)
mismatch_fp32_accum = []
# Scenario B: npu_ce with bf16 computation (pessimistic)
mismatch_bf16_comp = []
# H200 reference: flash_attn triton CE (fp32 accum) vs vLLM fp32
mismatch_h200 = []

for _ in range(N_SAMPLES):
    logits_fp32 = torch.randn(1, VOCAB_SIZE, dtype=torch.float32)
    label = torch.randint(0, VOCAB_SIZE, (1,))
    logits_bf16 = logits_fp32.to(torch.bfloat16)

    # vLLM inference (both platforms): bf16 -> fp32 log_softmax
    lp_vllm = F.log_softmax(logits_bf16.float(), dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()

    # Scenario A: npu_ce with fp32 accumulation
    lp_npu_fp32 = F.log_softmax(logits_bf16.float(), dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()
    mismatch_fp32_accum.append(lp_npu_fp32 - lp_vllm)

    # Scenario B: npu_ce with bf16 computation
    lp_npu_bf16 = F.log_softmax(logits_bf16, dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).float().item()
    mismatch_bf16_comp.append(lp_npu_bf16 - lp_vllm)

    # H200: flash_attn triton CE (fp32 accum) vs vLLM fp32
    lp_h200 = F.log_softmax(logits_bf16.float(), dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()
    mismatch_h200.append(lp_h200 - lp_vllm)

mismatch_fp32_accum = torch.tensor(mismatch_fp32_accum)
mismatch_bf16_comp = torch.tensor(mismatch_bf16_comp)
mismatch_h200 = torch.tensor(mismatch_h200)

eps_low, eps_high = 0.0003, 0.0004
ln_low = torch.log(torch.tensor(1 - eps_low)).item()
ln_high = torch.log(torch.tensor(1 + eps_high)).item()

print("=" * 70)
print("Train-vs-Inference mismatch analysis")
print(f"GSPO clip range in log-space: [{ln_low:.6f}, {ln_high:.6f}]")
print("=" * 70)
print()

for name, data in [
    ("H200 (flash_attn triton CE, fp32 accum)", mismatch_h200),
    ("NPU Scenario A (npu_ce, fp32 accum)", mismatch_fp32_accum),
    ("NPU Scenario B (npu_ce, bf16 comp)", mismatch_bf16_comp),
]:
    clipped = ((data < ln_low) | (data > ln_high)).float().mean().item()
    print(f"{name}:")
    print(f"  mean abs mismatch: {data.abs().mean().item():.8f}")
    print(f"  std:               {data.std().item():.8f}")
    print(f"  max abs:           {data.abs().max().item():.8f}")
    print(f"  % tokens clipped:  {clipped*100:.1f}%")
    print()

print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()
print("If npu_cross_entropy_loss does fp32 accumulation (Scenario A):")
print("  -> Same as H200, no train-vs-inference mismatch")
print("  -> The precision gap must come from elsewhere")
print()
print("If npu_cross_entropy_loss does bf16 computation (Scenario B):")
print(f"  -> {((mismatch_bf16_comp < ln_low) | (mismatch_bf16_comp > ln_high)).float().mean().item()*100:.0f}% of tokens clipped by GSPO due to numerical noise")
print("  -> This IS the smoking gun for the NPU precision gap")
print()
print("To determine which scenario is real, we need to run the NPU test")
print("(verify_npu_ce_precision.py) when NPU is available.")
print()
print("However, the documentation says output dtype = input dtype (bf16),")
print("which STRONGLY suggests Scenario B (bf16 computation).")
print("Most hardware-optimized fused kernels keep computation in the input dtype")
print("for performance, unlike the triton kernel which explicitly upcasts.")
