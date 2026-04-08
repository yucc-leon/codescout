#!/usr/bin/env python3
"""
验证 NPU 上 logprobs 计算的一致性。

测试：同一个模型、同一个输入，两次 forward 的 logprobs 是否 bit-exact。
如果不是，差异有多大？是否会影响 GSPO 的 clip range。

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0 python verify_logprobs_consistency.py
"""
import torch
import torch_npu
import torch.nn.functional as F

torch.npu.set_device(0)

model_path = "/sharedata/liyuchen/models/Qwen3-4B-Instruct-2507"

from transformers import AutoModelForCausalLM
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("npu")
model.eval()

S = 4096
V = model.config.vocab_size
input_ids = torch.randint(0, V, (1, S), device="npu")
labels = torch.roll(input_ids, -1, dims=1)

print(f"Testing logprobs consistency (S={S})...")

# Run forward twice
with torch.no_grad():
    logits1 = model(input_ids).logits
    logits2 = model(input_ids).logits

# Check logits consistency
logits_diff = (logits1 - logits2).abs()
print(f"\nLogits (bf16):")
print(f"  max diff: {logits_diff.max().item():.6f}")
print(f"  mean diff: {logits_diff.mean().item():.6f}")
print(f"  bit-exact: {torch.equal(logits1, logits2)}")

# Check logprobs consistency (fp32 upcast, like SkyRL)
with torch.no_grad():
    lp1 = F.log_softmax(logits1[0].float(), dim=-1).gather(-1, labels[0].unsqueeze(-1)).squeeze(-1)
    lp2 = F.log_softmax(logits2[0].float(), dim=-1).gather(-1, labels[0].unsqueeze(-1)).squeeze(-1)

lp_diff = (lp1 - lp2).abs()
print(f"\nLogprobs (fp32 log_softmax):")
print(f"  max diff: {lp_diff.max().item():.8f}")
print(f"  mean diff: {lp_diff.mean().item():.8f}")
print(f"  bit-exact: {torch.equal(lp1, lp2)}")

# What does this mean for GSPO?
gspo_clip_range = 0.0004 + 0.0003  # eps_clip_high + eps_clip_low
print(f"\nGSPO clip range: {gspo_clip_range}")
print(f"logprobs max diff / clip range: {lp_diff.max().item() / gspo_clip_range:.2f}x")

if lp_diff.max().item() > gspo_clip_range * 0.1:
    print("⚠ logprobs 不一致性超过 GSPO clip range 的 10%，可能影响训练稳定性")
else:
    print("✓ logprobs 一致性在 GSPO clip range 的 10% 以内")

# Also test: does NPU bf16 matmul have non-determinism?
print(f"\n--- NPU bf16 matmul determinism test ---")
A = torch.randn(1, 4096, 2560, dtype=torch.bfloat16, device="npu")
W = torch.randn(151936, 2560, dtype=torch.bfloat16, device="npu")
r1 = F.linear(A, W)
r2 = F.linear(A, W)
matmul_diff = (r1 - r2).abs()
print(f"matmul max diff: {matmul_diff.max().item():.6f}")
print(f"matmul bit-exact: {torch.equal(r1, r2)}")
