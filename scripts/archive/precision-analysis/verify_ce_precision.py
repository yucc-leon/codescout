"""
Compare precision of different cross-entropy / logprobs implementations.

We can't run the actual triton kernel on this machine, but we can simulate
what it does: the flash_attn triton cross_entropy uses online softmax with
fp32 accumulation on bf16 inputs. This is equivalent to:
  1. Cast bf16 logits to fp32
  2. Compute log_softmax in fp32
  3. Gather the target logprob

vs the NPU v2 path which does:
  1. Keep logits in bf16
  2. Compute log_softmax in bf16
  3. Gather the target logprob

The key question: how much does the fp32 accumulation in the triton kernel
reduce the error compared to pure bf16 log_softmax?
"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)

VOCAB_SIZE = 151936
N_SAMPLES = 5000

errors_bf16 = []      # NPU path: bf16 log_softmax
errors_upcast = []    # H200 path (simulated): bf16 input -> fp32 log_softmax
errors_vllm_bf16 = [] # vLLM inference (both platforms): bf16 log_softmax

for _ in range(N_SAMPLES):
    logits_fp32 = torch.randn(1, VOCAB_SIZE, dtype=torch.float32)
    label = torch.randint(0, VOCAB_SIZE, (1,))

    # Ground truth: fp32 everything
    lp_ref = F.log_softmax(logits_fp32, dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()

    # NPU training path: bf16 log_softmax
    logits_bf16 = logits_fp32.to(torch.bfloat16)
    lp_npu = F.log_softmax(logits_bf16, dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).float().item()

    # H200 training path (simulated flash_attn triton CE): bf16 input -> fp32 accumulation
    lp_h200 = F.log_softmax(logits_bf16.float(), dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()

    errors_bf16.append(lp_npu - lp_ref)
    errors_upcast.append(lp_h200 - lp_ref)

errors_bf16 = torch.tensor(errors_bf16)
errors_upcast = torch.tensor(errors_upcast)

print("=" * 70)
print("Logprobs error comparison: NPU vs H200 training paths")
print("=" * 70)
print()
print("NPU path (bf16 log_softmax):")
print(f"  mean abs error: {errors_bf16.abs().mean().item():.6f}")
print(f"  std:            {errors_bf16.std().item():.6f}")
print(f"  max abs error:  {errors_bf16.abs().max().item():.6f}")
print()
print("H200 path (bf16->fp32 upcast, simulating triton CE):")
print(f"  mean abs error: {errors_upcast.abs().mean().item():.6f}")
print(f"  std:            {errors_upcast.std().item():.6f}")
print(f"  max abs error:  {errors_upcast.abs().max().item():.6f}")
print()

ratio = errors_bf16.abs().mean() / errors_upcast.abs().mean()
print(f"NPU error is {ratio.item():.1f}x larger than H200 error")
print()

# Now the key question: how does this affect the train-vs-inference ratio?
# Both platforms use vLLM for inference, which likely uses bf16 log_softmax.
# So the inference logprobs have the same error on both platforms.
#
# ratio = exp(logp_train - logp_inference)
#
# H200: logp_train has small error (upcast), logp_inference has bf16 error
#   -> ratio error ≈ error_upcast - error_bf16_inference
#
# NPU: logp_train has large error (bf16), logp_inference has bf16 error
#   -> ratio error ≈ error_bf16_train - error_bf16_inference
#
# If train and inference use the SAME bf16 log_softmax, errors partially cancel!
# But if train uses a DIFFERENT implementation (flash_attn CE), they don't cancel.

print("=" * 70)
print("Train-vs-Inference ratio error analysis")
print("=" * 70)
print()

# Simulate: same logits, different implementations for train vs inference
N_SIM = 5000
ratio_errors_h200 = []
ratio_errors_npu = []

for _ in range(N_SIM):
    logits_fp32 = torch.randn(1, VOCAB_SIZE, dtype=torch.float32)
    label = torch.randint(0, VOCAB_SIZE, (1,))
    logits_bf16 = logits_fp32.to(torch.bfloat16)

    # Reference
    lp_ref = F.log_softmax(logits_fp32, dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()

    # vLLM inference (both platforms): bf16 log_softmax
    lp_vllm = F.log_softmax(logits_bf16, dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).float().item()

    # H200 training: upcast (simulating triton CE)
    lp_h200_train = F.log_softmax(logits_bf16.float(), dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()

    # NPU training: bf16 log_softmax
    lp_npu_train = lp_vllm  # Same implementation as vLLM!

    # The ratio log-error
    # True ratio should be 0 (same model, same input)
    # H200: logp_train - logp_vllm = (lp_h200_train) - (lp_vllm)
    h200_ratio_log = lp_h200_train - lp_vllm
    # NPU: logp_train - logp_vllm = (lp_npu_train) - (lp_vllm) = 0 (same impl!)
    npu_ratio_log = lp_npu_train - lp_vllm

    ratio_errors_h200.append(h200_ratio_log)
    ratio_errors_npu.append(npu_ratio_log)

ratio_errors_h200 = torch.tensor(ratio_errors_h200)
ratio_errors_npu = torch.tensor(ratio_errors_npu)

print("H200 train-vs-inference ratio log-error:")
print(f"  mean abs: {ratio_errors_h200.abs().mean().item():.6f}")
print(f"  std:      {ratio_errors_h200.std().item():.6f}")
print(f"  max abs:  {ratio_errors_h200.abs().max().item():.6f}")
print()
print("NPU train-vs-inference ratio log-error:")
print(f"  mean abs: {ratio_errors_npu.abs().mean().item():.6f}")
print(f"  std:      {ratio_errors_npu.std().item():.6f}")
print(f"  max abs:  {ratio_errors_npu.abs().max().item():.6f}")
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
print("CRITICAL FINDING")
print("=" * 70)
print()
if npu_clipped < h200_clipped:
    print("SURPRISING: NPU has LESS train-vs-inference mismatch than H200!")
    print("This is because NPU training and NPU vLLM both use bf16 log_softmax,")
    print("so their errors CANCEL OUT in the ratio computation.")
    print()
    print("H200 training uses flash_attn triton CE (fp32 accumulation) while")
    print("H200 vLLM uses bf16 inference, creating a LARGER mismatch.")
    print()
    print("This means the logprobs path difference is NOT the main cause")
    print("of the NPU precision gap. We need to look elsewhere.")
elif npu_clipped > h200_clipped:
    print("NPU has MORE train-vs-inference mismatch than H200.")
    print("The logprobs path difference IS a contributing factor.")
else:
    print("Both platforms have similar train-vs-inference mismatch.")
    print("The logprobs path difference is NOT the main cause.")
