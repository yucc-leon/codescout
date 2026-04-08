"""
Now that we confirmed npu_cross_entropy_loss = bf16 log_softmax precision,
let's quantify the ACTUAL fix impact more carefully.

The 69.1% clip rate for fp32 log_softmax seems high. But remember:
- vLLM computes logprobs on bf16 logits upcast to fp32
- If training also does bf16 logits upcast to fp32, they should be IDENTICAL
- The 69.1% might be from the fp32 reference being computed on fp32 logits
  (not bf16 logits upcast to fp32)

Let me verify: if both train and vLLM start from the SAME bf16 logits
and both upcast to fp32, the mismatch should be exactly 0.
"""

import torch
import torch_npu
import torch.nn.functional as F

torch.npu.set_device(0)
torch.manual_seed(42)

VOCAB_SIZE = 151936
N_SAMPLES = 1000

eps_low, eps_high = 0.0003, 0.0004
ln_low = torch.log(torch.tensor(1 - eps_low)).item()
ln_high = torch.log(torch.tensor(1 + eps_high)).item()

# Test 1: Both use fp32 log_softmax on same bf16 logits -> should be 0 mismatch
mismatch_same_impl = []
# Test 2: npu_ce (bf16) vs fp32 log_softmax -> current NPU mismatch
mismatch_current = []
# Test 3: fp32 npu_ce vs fp32 log_softmax -> proposed fix
mismatch_fix = []

for _ in range(N_SAMPLES):
    logits_fp32_cpu = torch.randn(1, VOCAB_SIZE, dtype=torch.float32)
    label_cpu = torch.randint(0, VOCAB_SIZE, (1,))

    logits_bf16_npu = logits_fp32_cpu.to(torch.bfloat16).npu()
    label_npu = label_cpu.npu()

    # vLLM: bf16 -> fp32 log_softmax (this is what vLLM does)
    lp_vllm = F.log_softmax(logits_bf16_npu.float(), dim=-1)
    lp_vllm_val = lp_vllm.gather(-1, label_npu.unsqueeze(-1)).squeeze(-1).cpu().float().item()

    # Test 1: Training also does bf16 -> fp32 log_softmax (identical to vLLM)
    lp_train_fp32 = F.log_softmax(logits_bf16_npu.float(), dim=-1)
    lp_train_fp32_val = lp_train_fp32.gather(-1, label_npu.unsqueeze(-1)).squeeze(-1).cpu().float().item()
    mismatch_same_impl.append(lp_train_fp32_val - lp_vllm_val)

    # Test 2: Current NPU training: npu_cross_entropy_loss on bf16
    loss_bf16, _, _, _ = torch_npu.npu_cross_entropy_loss(logits_bf16_npu, label_npu, reduction="none")
    lp_current = -loss_bf16.cpu().float().item()
    mismatch_current.append(lp_current - lp_vllm_val)

    # Test 3: Proposed fix: npu_cross_entropy_loss on fp32
    loss_fp32, _, _, _ = torch_npu.npu_cross_entropy_loss(logits_bf16_npu.float(), label_npu, reduction="none")
    lp_fix = -loss_fp32.cpu().float().item()
    mismatch_fix.append(lp_fix - lp_vllm_val)

mismatch_same_impl = torch.tensor(mismatch_same_impl)
mismatch_current = torch.tensor(mismatch_current)
mismatch_fix = torch.tensor(mismatch_fix)

print("=" * 70)
print("Train-vs-Inference mismatch on NPU")
print(f"GSPO clip range: [{ln_low:.6f}, {ln_high:.6f}]")
print("=" * 70)

for name, data in [
    ("Same impl (fp32 log_softmax both sides)", mismatch_same_impl),
    ("Current (npu_ce bf16 vs vLLM fp32)", mismatch_current),
    ("Fix A (npu_ce fp32 vs vLLM fp32)", mismatch_fix),
]:
    clipped = ((data < ln_low) | (data > ln_high)).float().mean().item()
    print(f"\n{name}:")
    print(f"  mean abs: {data.abs().mean().item():.8f}")
    print(f"  std:      {data.std().item():.8f}")
    print(f"  max abs:  {data.abs().max().item():.8f}")
    print(f"  % clipped by GSPO: {clipped*100:.1f}%")
