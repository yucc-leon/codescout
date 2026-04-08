#!/usr/bin/env python3
"""Compare sdpa vs flash_attention_2 logits on NPU."""
import torch, torch_npu, gc
torch.npu.set_device(0)
from transformers import AutoModelForCausalLM

S = 128
input_ids = torch.randint(0, 1000, (1, S), device="npu")
attn_mask = torch.ones(1, S, dtype=torch.long, device="npu")
attn_mask[0, -20:] = 0

path = "/sharedata/liyuchen/models/Qwen3-4B-Instruct-2507"

# SDPA
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").to("npu")
m.eval()
with torch.no_grad():
    o1 = m(input_ids, attention_mask=attn_mask).logits.cpu()
del m; gc.collect(); torch.npu.empty_cache()

# flash_attention_2
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("npu")
m.eval()
with torch.no_grad():
    o2 = m(input_ids, attention_mask=attn_mask).logits.cpu()
del m; gc.collect(); torch.npu.empty_cache()

diff = (o1 - o2).abs()
v = attn_mask[0].bool().cpu()
print(f"Non-pad: max={diff[0,v].max():.6f} mean={diff[0,v].mean():.6f}")
print(f"Pad:     max={diff[0,~v].max():.6f} mean={diff[0,~v].mean():.6f}")
print(f"Bit-exact: {torch.equal(o1, o2)}")
