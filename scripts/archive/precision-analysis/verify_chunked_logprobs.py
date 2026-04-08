#!/usr/bin/env python3
"""Verify chunked_logprobs_from_hidden_states matches the original logprobs_from_logits."""
import torch
import torch_npu
import sys
sys.path.insert(0, "SkyRL/skyrl-train")

from skyrl_train.utils.torch_utils import (
    logprobs_from_logits_v2,
    chunked_logprobs_from_hidden_states,
    chunked_entropy_from_hidden_states,
    chunked_entropy_from_logits,
)

torch.npu.set_device(0)
device = "npu"
dtype = torch.bfloat16

B, S, H, V = 1, 4096, 2560, 151643
print(f"Testing B={B}, S={S}, H={H}, V={V}")

# Simulate hidden_states and lm_head
hidden = torch.randn(B, S, H, dtype=dtype, device=device)
lm_head = torch.nn.Linear(H, V, bias=False, dtype=dtype, device=device)
labels = torch.randint(0, V, (B, S), device=device)

# Original: full logits -> logprobs
with torch.no_grad():
    logits = lm_head(hidden)
    logprobs_orig = logprobs_from_logits_v2(logits, labels)
    mem_orig = torch.npu.max_memory_allocated() / 1e9

torch.npu.reset_peak_memory_stats()

# Chunked: hidden_states -> chunked logprobs
with torch.no_grad():
    logprobs_chunked = chunked_logprobs_from_hidden_states(hidden, labels, lm_head, temperature=1.0)
    mem_chunked = torch.npu.max_memory_allocated() / 1e9

# Compare
diff = (logprobs_orig - logprobs_chunked).abs()
print(f"logprobs max diff: {diff.max().item():.6f}")
print(f"logprobs mean diff: {diff.mean().item():.6f}")
print(f"Memory: original={mem_orig:.2f} GB, chunked={mem_chunked:.2f} GB, saved={mem_orig - mem_chunked:.2f} GB")

# Verify entropy too
torch.npu.reset_peak_memory_stats()
with torch.no_grad():
    entropy_orig = chunked_entropy_from_logits(logits, requires_grad=False)
    mem_ent_orig = torch.npu.max_memory_allocated() / 1e9

torch.npu.reset_peak_memory_stats()
with torch.no_grad():
    entropy_chunked = chunked_entropy_from_hidden_states(hidden, lm_head, temperature=1.0, requires_grad=False)
    mem_ent_chunked = torch.npu.max_memory_allocated() / 1e9

ent_diff = (entropy_orig - entropy_chunked).abs()
print(f"\nentropy max diff: {ent_diff.max().item():.6f}")
print(f"entropy mean diff: {ent_diff.mean().item():.6f}")
print(f"Memory: original={mem_ent_orig:.2f} GB, chunked={mem_ent_chunked:.2f} GB")

print("\n✓ All checks passed" if diff.max().item() < 0.05 and ent_diff.max().item() < 0.01 else "\n✗ MISMATCH")
