#!/usr/bin/env python3
"""Test whether sample_packing causes cross-sequence attention leakage on NPU."""
import npu_support.patch_cuda
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/sharedata/liyuchen/models/Qwen3-4B-Instruct-2507"
device = "npu:0"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="flash_attention_2"
).to(device).eval()

text1 = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n" * 10
text2 = "def hello():\n    print('hello world')\n" * 8

ids1 = tokenizer.encode(text1, add_special_tokens=False)[:512]
ids2 = tokenizer.encode(text2, add_special_tokens=False)[:300]
pad_id = tokenizer.pad_token_id
max_len = max(len(ids1), len(ids2))

# Batched with padding (B=2)
padded2 = [pad_id] * (max_len - len(ids2)) + ids2
sequences = torch.tensor([ids1, padded2], device=device)
attn_mask = torch.tensor([
    [1]*len(ids1),
    [0]*(max_len - len(ids2)) + [1]*len(ids2),
], device=device)
pos_ids = attn_mask.long().cumsum(-1) - 1
pos_ids.masked_fill_(attn_mask == 0, 1)

with torch.no_grad():
    out_batched = model(sequences, attention_mask=attn_mask, position_ids=pos_ids)

# Packed (both into one)
from flash_attn.bert_padding import unpad_input, pad_input

packed_seq, nnz_indices, _, _, _ = unpad_input(
    sequences.unsqueeze(-1), attention_mask=attn_mask
)
packed_seq = packed_seq.transpose(0, 1)
packed_pos, _, _, _, _ = unpad_input(pos_ids.unsqueeze(-1), attn_mask)
packed_pos = packed_pos.transpose(0, 1)

print(f"Packed shape: {packed_seq.shape} (expected (1, {len(ids1)+len(ids2)}))")
print(f"Position IDs around boundary: {packed_pos[0,len(ids1)-3:len(ids1)+3].tolist()}")

with torch.no_grad():
    out_packed = model(packed_seq, attention_mask=None, position_ids=packed_pos)

# Unpack
logits_unpacked = pad_input(
    out_packed.logits.squeeze(0), indices=nnz_indices, batch=2, seqlen=max_len
)

# Compare seq 1 (no padding, full length)
diff1 = (out_batched.logits[0].float() - logits_unpacked[0].float()).abs()
print(f"\nSeq 1 (len={len(ids1)}, no padding):")
print(f"  Max logit diff: {diff1.max().item():.8f}")
print(f"  Mean logit diff: {diff1.mean().item():.8f}")

# Compare seq 2 (non-padding positions only)
s2_start = max_len - len(ids2)
diff2 = (out_batched.logits[1, s2_start:].float() - logits_unpacked[1, s2_start:].float()).abs()
print(f"\nSeq 2 (len={len(ids2)}, non-pad positions):")
print(f"  Max logit diff: {diff2.max().item():.8f}")
print(f"  Mean logit diff: {diff2.mean().item():.8f}")

# Also test: single seq (B=1) vs packed
single_ids1 = torch.tensor([ids1], device=device)
single_mask1 = torch.ones_like(single_ids1)
single_pos1 = torch.arange(len(ids1), device=device).unsqueeze(0)
with torch.no_grad():
    out_single = model(single_ids1, attention_mask=single_mask1, position_ids=single_pos1)

diff_sv_b = (out_single.logits[0].float() - out_batched.logits[0].float()).abs()
diff_sv_p = (out_single.logits[0].float() - logits_unpacked[0].float()).abs()
print(f"\nSeq 1: single(B=1) vs batched(B=2): max={diff_sv_b.max().item():.8f}")
print(f"Seq 1: single(B=1) vs packed:      max={diff_sv_p.max().item():.8f}")

if diff1.max().item() > 0.01 or diff2.max().item() > 0.01:
    print("\nWARNING: PACKING CAUSES CROSS-SEQUENCE ATTENTION LEAKAGE ON NPU!")
    print("The varlen attention is NOT properly isolating sequences.")
else:
    print("\nOK: Packing properly isolates sequences.")
    print("The logprob diff in the previous test was from padding handling.")
