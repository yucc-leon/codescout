"""
Test vLLM inference determinism on NPU.

If the same prompt with temperature=0 produces different outputs across runs,
or if NPU inference diverges from what we'd expect from the model,
this could explain the reward gap.

We test:
1. Is NPU vLLM inference deterministic (same output for same input)?
2. Does the base model (no training) produce reasonable outputs?
3. Compare logprobs distribution between NPU eager attention and NPU SDPA
"""

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)
torch.npu.set_device(0)

MODEL_PATH = "/sharedata/liyuchen/models/Qwen3-4B-Instruct-2507"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Test prompt - a simple code search task
prompt = """<|im_start|>system
You are a code search agent. Find the relevant files.<|im_end|>
<|im_start|>user
Find the file that handles user authentication in a Django project.<|im_end|>
<|im_start|>assistant
"""

input_ids = tokenizer.encode(prompt, return_tensors="pt").npu()

print(f"Input length: {input_ids.shape[1]} tokens")
print()

# Test 1: Eager attention (what NPU training uses with flash_attn=false)
print("=" * 60)
print("Test 1: Eager attention (attn_implementation='eager')")
print("=" * 60)
model_eager = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).npu().eval()

with torch.no_grad():
    out1 = model_eager(input_ids)
    logits1 = out1.logits[:, -1, :]  # last token logits
    top5_1 = torch.topk(logits1.float(), 5, dim=-1)
    
    # Run again to check determinism
    out2 = model_eager(input_ids)
    logits2 = out2.logits[:, -1, :]
    
    diff = (logits1.float() - logits2.float()).abs()
    print(f"Determinism check (same model, same input):")
    print(f"  max logit diff: {diff.max().item():.8f}")
    print(f"  mean logit diff: {diff.mean().item():.8f}")
    print(f"  (should be 0.0 for deterministic inference)")
    print()
    
    print(f"Top-5 next tokens (eager):")
    for i in range(5):
        token_id = top5_1.indices[0, i].item()
        logit = top5_1.values[0, i].item()
        token = tokenizer.decode([token_id])
        print(f"  {i+1}. '{token}' (id={token_id}, logit={logit:.4f})")

del model_eager
torch.npu.empty_cache()

# Test 2: SDPA attention (what NPU might use internally)
print()
print("=" * 60)
print("Test 2: SDPA attention (attn_implementation='sdpa')")
print("=" * 60)
try:
    model_sdpa = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).npu().eval()

    with torch.no_grad():
        out3 = model_sdpa(input_ids)
        logits3 = out3.logits[:, -1, :]
        top5_3 = torch.topk(logits3.float(), 5, dim=-1)
        
        print(f"Top-5 next tokens (sdpa):")
        for i in range(5):
            token_id = top5_3.indices[0, i].item()
            logit = top5_3.values[0, i].item()
            token = tokenizer.decode([token_id])
            print(f"  {i+1}. '{token}' (id={token_id}, logit={logit:.4f})")
        
        # Compare eager vs sdpa
        diff_attn = (logits1.float() - logits3.float()).abs()
        print(f"\nEager vs SDPA logit difference:")
        print(f"  max: {diff_attn.max().item():.6f}")
        print(f"  mean: {diff_attn.mean().item():.6f}")
        
        # Check if top-1 token is the same
        top1_eager = top5_1.indices[0, 0].item()
        top1_sdpa = top5_3.indices[0, 0].item()
        print(f"  Top-1 token match: {top1_eager == top1_sdpa} (eager={top1_eager}, sdpa={top1_sdpa})")
    
    del model_sdpa
    torch.npu.empty_cache()
except Exception as e:
    print(f"SDPA not available on NPU: {e}")

# Test 3: Generate a short sequence and check consistency
print()
print("=" * 60)
print("Test 3: Greedy generation (temperature=0 equivalent)")
print("=" * 60)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).npu().eval()

with torch.no_grad():
    gen1 = model.generate(input_ids, max_new_tokens=50, do_sample=False, temperature=1.0)
    gen2 = model.generate(input_ids, max_new_tokens=50, do_sample=False, temperature=1.0)
    
    tokens1 = gen1[0, input_ids.shape[1]:].tolist()
    tokens2 = gen2[0, input_ids.shape[1]:].tolist()
    
    match = tokens1 == tokens2
    print(f"Greedy generation deterministic: {match}")
    if not match:
        for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
            if t1 != t2:
                print(f"  First divergence at position {i}: {t1} vs {t2}")
                print(f"  '{tokenizer.decode([t1])}' vs '{tokenizer.decode([t2])}'")
                break
    
    text1 = tokenizer.decode(tokens1, skip_special_tokens=True)
    print(f"\nGenerated text (first 200 chars):")
    print(f"  {text1[:200]}")

del model
torch.npu.empty_cache()
print("\nDone.")
