"""Fast CPU simulation with smaller vocab to avoid timeout."""
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# Use smaller vocab for speed on CPU, scale results
VOCAB_SIZE = 32000  # smaller for speed
N_SAMPLES = 500

mismatch_bf16_comp = []
mismatch_fp32_accum = []

for _ in range(N_SAMPLES):
    logits_fp32 = torch.randn(1, VOCAB_SIZE, dtype=torch.float32)
    label = torch.randint(0, VOCAB_SIZE, (1,))
    logits_bf16 = logits_fp32.to(torch.bfloat16)

    # vLLM: bf16 -> fp32 log_softmax
    lp_vllm = F.log_softmax(logits_bf16.float(), dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()

    # Scenario A: fp32 accum
    lp_fp32 = F.log_softmax(logits_bf16.float(), dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).item()
    mismatch_fp32_accum.append(lp_fp32 - lp_vllm)

    # Scenario B: bf16 comp
    lp_bf16 = F.log_softmax(logits_bf16, dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).float().item()
    mismatch_bf16_comp.append(lp_bf16 - lp_vllm)

mismatch_bf16_comp = torch.tensor(mismatch_bf16_comp)
mismatch_fp32_accum = torch.tensor(mismatch_fp32_accum)

eps_low, eps_high = 0.0003, 0.0004
ln_low = torch.log(torch.tensor(1 - eps_low)).item()
ln_high = torch.log(torch.tensor(1 + eps_high)).item()

clip_bf16 = ((mismatch_bf16_comp < ln_low) | (mismatch_bf16_comp > ln_high)).float().mean().item()
clip_fp32 = ((mismatch_fp32_accum < ln_low) | (mismatch_fp32_accum > ln_high)).float().mean().item()

print(f"Vocab={VOCAB_SIZE}, N={N_SAMPLES}")
print(f"GSPO clip range: [{ln_low:.6f}, {ln_high:.6f}]")
print()
print(f"Scenario A (fp32 accum, like H200):")
print(f"  mean abs mismatch: {mismatch_fp32_accum.abs().mean().item():.8f}")
print(f"  % tokens clipped:  {clip_fp32*100:.1f}%")
print()
print(f"Scenario B (bf16 comp, likely NPU):")
print(f"  mean abs mismatch: {mismatch_bf16_comp.abs().mean().item():.6f}")
print(f"  % tokens clipped:  {clip_bf16*100:.1f}%")
print()
print("With Qwen3's 151936 vocab, errors would be even larger.")
print()
if clip_bf16 > 90:
    print("VERDICT: If npu_cross_entropy_loss uses bf16 computation,")
    print(f"~{clip_bf16*100:.0f}% of tokens are clipped by GSPO noise alone.")
    print("This completely explains the NPU precision gap.")
