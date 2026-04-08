"""
Test the precision of torch_npu.npu_cross_entropy_loss vs fp32 reference.

This is the ACTUAL function used on NPU for logprobs computation in verl.
We need to know if it does fp32 accumulation internally or not.
"""

import torch
import torch_npu
import torch.nn.functional as F

torch.manual_seed(42)
torch.npu.set_device(0)

VOCAB_SIZE = 151936  # Qwen3 vocab
N_SAMPLES = 1000

errors_npu_ce = []
errors_bf16_logsoftmax = []
errors_fp32_logsoftmax = []

for i in range(N_SAMPLES):
    # Generate random logits
    logits_fp32_cpu = torch.randn(1, VOCAB_SIZE, dtype=torch.float32)
    label_cpu = torch.randint(0, VOCAB_SIZE, (1,))

    # fp32 reference on CPU
    lp_ref = F.log_softmax(logits_fp32_cpu, dim=-1).gather(-1, label_cpu.unsqueeze(-1)).squeeze(-1).item()

    # bf16 on NPU
    logits_bf16_npu = logits_fp32_cpu.to(torch.bfloat16).npu()
    label_npu = label_cpu.npu()

    # Path 1: torch_npu.npu_cross_entropy_loss (what verl actually uses)
    loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(
        logits_bf16_npu, label_npu, reduction="none"
    )
    lp_npu_ce = -loss.cpu().float().item()

    # Path 2: bf16 log_softmax on NPU (what v2 fallback would do)
    lp_bf16 = F.log_softmax(logits_bf16_npu, dim=-1).gather(-1, label_npu.unsqueeze(-1)).squeeze(-1).cpu().float().item()

    # Path 3: upcast to fp32 on NPU then log_softmax (ideal)
    lp_fp32_npu = F.log_softmax(logits_bf16_npu.float(), dim=-1).gather(-1, label_npu.unsqueeze(-1)).squeeze(-1).cpu().float().item()

    errors_npu_ce.append(lp_npu_ce - lp_ref)
    errors_bf16_logsoftmax.append(lp_bf16 - lp_ref)
    errors_fp32_logsoftmax.append(lp_fp32_npu - lp_ref)

    if i < 3:
        print(f"Sample {i}: ref={lp_ref:.6f}, npu_ce={lp_npu_ce:.6f}, bf16={lp_bf16:.6f}, fp32_npu={lp_fp32_npu:.6f}")

errors_npu_ce = torch.tensor(errors_npu_ce)
errors_bf16_logsoftmax = torch.tensor(errors_bf16_logsoftmax)
errors_fp32_logsoftmax = torch.tensor(errors_fp32_logsoftmax)

print()
print("=" * 70)
print("Precision comparison of logprobs computation paths")
print("=" * 70)
print()
print("1. torch_npu.npu_cross_entropy_loss (ACTUAL NPU training path):")
print(f"   mean abs error: {errors_npu_ce.abs().mean().item():.6f}")
print(f"   std:            {errors_npu_ce.std().item():.6f}")
print(f"   max abs error:  {errors_npu_ce.abs().max().item():.6f}")
print()
print("2. bf16 F.log_softmax on NPU (v2 fallback, NOT used):")
print(f"   mean abs error: {errors_bf16_logsoftmax.abs().mean().item():.6f}")
print(f"   std:            {errors_bf16_logsoftmax.std().item():.6f}")
print(f"   max abs error:  {errors_bf16_logsoftmax.abs().max().item():.6f}")
print()
print("3. fp32 F.log_softmax on NPU (ideal, matches vLLM):")
print(f"   mean abs error: {errors_fp32_logsoftmax.abs().mean().item():.6f}")
print(f"   std:            {errors_fp32_logsoftmax.std().item():.6f}")
print(f"   max abs error:  {errors_fp32_logsoftmax.abs().max().item():.6f}")
print()

# Compare with GSPO clip range
eps_low, eps_high = 0.0003, 0.0004
ln_low = torch.log(torch.tensor(1 - eps_low)).item()
ln_high = torch.log(torch.tensor(1 + eps_high)).item()

npu_ce_clipped = ((errors_npu_ce < ln_low) | (errors_npu_ce > ln_high)).float().mean().item()
bf16_clipped = ((errors_bf16_logsoftmax < ln_low) | (errors_bf16_logsoftmax > ln_high)).float().mean().item()
fp32_clipped = ((errors_fp32_logsoftmax < ln_low) | (errors_fp32_logsoftmax > ln_high)).float().mean().item()

print(f"GSPO clip range: [{1-eps_low}, {1+eps_high}]")
print(f"Tokens clipped by numerical noise alone:")
print(f"  npu_cross_entropy_loss: {npu_ce_clipped*100:.1f}%")
print(f"  bf16 log_softmax:      {bf16_clipped*100:.1f}%")
print(f"  fp32 log_softmax:      {fp32_clipped*100:.1f}%")
print()

# Now compare with vLLM's fp32 logprobs (the actual train-vs-inference mismatch)
print("=" * 70)
print("Train-vs-Inference mismatch (train logprobs vs vLLM fp32 logprobs)")
print("=" * 70)
print()

mismatch_npu_ce = []
mismatch_bf16 = []

for i in range(N_SAMPLES):
    logits_fp32_cpu = torch.randn(1, VOCAB_SIZE, dtype=torch.float32)
    label_cpu = torch.randint(0, VOCAB_SIZE, (1,))
    logits_bf16_npu = logits_fp32_cpu.to(torch.bfloat16).npu()
    label_npu = label_cpu.npu()

    # vLLM inference: bf16 logits -> fp32 log_softmax
    lp_vllm = F.log_softmax(logits_bf16_npu.float(), dim=-1).gather(-1, label_npu.unsqueeze(-1)).squeeze(-1).cpu().float().item()

    # NPU training: npu_cross_entropy_loss
    loss, _, _, _ = torch_npu.npu_cross_entropy_loss(logits_bf16_npu, label_npu, reduction="none")
    lp_train = -loss.cpu().float().item()

    mismatch_npu_ce.append(lp_train - lp_vllm)

    # For comparison: bf16 log_softmax
    lp_bf16 = F.log_softmax(logits_bf16_npu, dim=-1).gather(-1, label_npu.unsqueeze(-1)).squeeze(-1).cpu().float().item()
    mismatch_bf16.append(lp_bf16 - lp_vllm)

mismatch_npu_ce = torch.tensor(mismatch_npu_ce)
mismatch_bf16 = torch.tensor(mismatch_bf16)

print("npu_cross_entropy_loss vs vLLM fp32 (ACTUAL mismatch):")
print(f"  mean abs: {mismatch_npu_ce.abs().mean().item():.6f}")
print(f"  std:      {mismatch_npu_ce.std().item():.6f}")
print(f"  max abs:  {mismatch_npu_ce.abs().max().item():.6f}")

npu_ce_clip = ((mismatch_npu_ce < ln_low) | (mismatch_npu_ce > ln_high)).float().mean().item()
print(f"  % tokens clipped by GSPO: {npu_ce_clip*100:.1f}%")
print()

print("bf16 log_softmax vs vLLM fp32 (hypothetical):")
print(f"  mean abs: {mismatch_bf16.abs().mean().item():.6f}")
print(f"  std:      {mismatch_bf16.std().item():.6f}")
print(f"  max abs:  {mismatch_bf16.abs().max().item():.6f}")

bf16_clip = ((mismatch_bf16 < ln_low) | (mismatch_bf16 > ln_high)).float().mean().item()
print(f"  % tokens clipped by GSPO: {bf16_clip*100:.1f}%")
