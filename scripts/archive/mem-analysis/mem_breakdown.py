#!/usr/bin/env python3
"""显存分解分析"""
S = 37000
H = 2560
I = 9728
bf16 = 2
V = 151643

attn_act = S * H * bf16 * 4 / 1e9
mlp_act = S * I * bf16 * 4 / 1e9
ckpt_per_layer = S * H * bf16 / 1e9
ckpt_total = ckpt_per_layer * 36
recompute_peak = attn_act + mlp_act

logits_gb = S * V * bf16 / 1e9
params_full = 4e9 * bf16 / 1e9
opt_shard_4 = 4e9 * 4 * 2 / 4 / 1e9
grad_shard_4 = 4e9 * bf16 / 4 / 1e9

total_4 = logits_gb * 2 + params_full + ckpt_total + recompute_peak + opt_shard_4 + grad_shard_4

print(f"=== glen=37000, FSDP 4 cards ===")
print(f"logits+logprobs:  {logits_gb*2:.1f} GB  (NOT sharded by FSDP)")
print(f"params(gathered): {params_full:.1f} GB")
print(f"ckpt activations: {ckpt_total:.1f} GB  (NOT sharded by FSDP)")
print(f"recompute peak:   {recompute_peak:.1f} GB")
print(f"optimizer(shard): {opt_shard_4:.1f} GB  (sharded)")
print(f"gradients(shard): {grad_shard_4:.1f} GB  (sharded)")
print(f"TOTAL:            {total_4:.1f} GB  (limit 64 GB)")
print(f"headroom:         {64 - total_4:.1f} GB")

opt_shard_8 = 4e9 * 4 * 2 / 8 / 1e9
grad_shard_8 = 4e9 * bf16 / 8 / 1e9
total_8 = logits_gb * 2 + params_full + ckpt_total + recompute_peak + opt_shard_8 + grad_shard_8

print(f"\n=== glen=37000, FSDP 8 cards ===")
print(f"optimizer(shard): {opt_shard_8:.1f} GB")
print(f"gradients(shard): {grad_shard_8:.1f} GB")
print(f"TOTAL:            {total_8:.1f} GB  (limit 64 GB)")
print(f"headroom:         {64 - total_8:.1f} GB")
print(f"Savings vs 4:     {total_4 - total_8:.1f} GB")

# What FSDP shards vs what it doesn't
sharded = opt_shard_4 + grad_shard_4
not_sharded = logits_gb * 2 + ckpt_total + recompute_peak
print(f"\n=== What FSDP shards vs doesn't ===")
print(f"Sharded (scales with cards):     {sharded:.1f} GB ({sharded/total_4*100:.0f}%)")
print(f"NOT sharded (fixed per card):    {not_sharded:.1f} GB ({not_sharded/total_4*100:.0f}%)")
print(f"params (gathered, temporary):    {params_full:.1f} GB")

print(f"\n=== Better solutions ===")
print(f"1. Chunked logprobs: don't materialize full (S, V) logits")
print(f"   Saves: {logits_gb:.1f} GB per intermediate")
print(f"2. Truncate max_response_len < 34000")
print(f"   At glen=33000: logits = {33000*V*bf16/1e9:.1f} GB (saves {(S-33000)*V*bf16/1e9:.1f} GB)")
print(f"3. Sequence parallel: shard activations along S dim")
print(f"   Saves: {ckpt_total:.1f} GB checkpoint + proportional logits")
