#!/usr/bin/env python3
"""
Quantify the NPU batch-dimension numerical inconsistency.

On NPU, model(x, B=1) != model(x, B=2) even with identical input and no padding.
This script isolates which layer(s) cause the divergence and measures the impact
on GSPO training logprobs.
"""
import npu_support.patch_cuda  # noqa
import argparse
import torch
import torch.nn.functional as F
import math


def test_raw_attention_kernel(device):
    """Test npu_fusion_attention directly with different batch sizes."""
    from torch_npu import npu_fusion_attention

    print("=" * 70)
    print("Test 1: Raw npu_fusion_attention kernel, B=1 vs B=2")
    print("=" * 70)

    torch.manual_seed(42)
    S, H, D = 128, 32, 128
    scale = 1.0 / math.sqrt(D)
    attn_mask = torch.triu(torch.ones([2048, 2048], device=device), diagonal=1).bool()

    q1 = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)
    k1 = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)
    v1 = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)

    out1 = npu_fusion_attention(q1, k1, v1, H, "BSND", keep_prob=1.0,
                                scale=scale, atten_mask=attn_mask, sparse_mode=3)[0]

    # B=2: duplicate the same data
    q2 = q1.expand(2, -1, -1, -1).contiguous()
    k2 = k1.expand(2, -1, -1, -1).contiguous()
    v2 = v1.expand(2, -1, -1, -1).contiguous()

    out2 = npu_fusion_attention(q2, k2, v2, H, "BSND", keep_prob=1.0,
                                scale=scale, atten_mask=attn_mask, sparse_mode=3)[0]

    diff = (out1[0].float() - out2[0].float()).abs()
    print(f"  Attention kernel B=1 vs B=2: max={diff.max():.6f}, mean={diff.mean():.8f}")
    if diff.max() < 1e-5:
        print("  => Kernel is batch-independent. Bug is elsewhere.\n")
    else:
        print("  => Kernel itself has batch-dependent behavior!\n")
    return diff.max().item()


def test_rmsnorm(device):
    """Test RMSNorm with different batch sizes."""
    print("=" * 70)
    print("Test 2: RMSNorm, B=1 vs B=2")
    print("=" * 70)

    torch.manual_seed(42)
    H = 2560  # Qwen3-4B hidden size
    S = 128

    # Create a simple RMSNorm
    norm = torch.nn.RMSNorm(H, dtype=torch.bfloat16).to(device)

    x1 = torch.randn(1, S, H, dtype=torch.bfloat16, device=device)
    out1 = norm(x1)

    x2 = x1.expand(2, -1, -1).contiguous()
    out2 = norm(x2)

    diff = (out1[0].float() - out2[0].float()).abs()
    print(f"  RMSNorm B=1 vs B=2: max={diff.max():.6f}, mean={diff.mean():.8f}")
    if diff.max() < 1e-5:
        print("  => RMSNorm is batch-independent.\n")
    else:
        print("  => RMSNorm has batch-dependent behavior!\n")
    return diff.max().item()


def test_linear(device):
    """Test nn.Linear with different batch sizes."""
    print("=" * 70)
    print("Test 3: nn.Linear (MLP), B=1 vs B=2")
    print("=" * 70)

    torch.manual_seed(42)
    H = 2560

    linear = torch.nn.Linear(H, H * 4, dtype=torch.bfloat16, bias=False).to(device)

    x1 = torch.randn(1, 128, H, dtype=torch.bfloat16, device=device)
    out1 = linear(x1)

    x2 = x1.expand(2, -1, -1).contiguous()
    out2 = linear(x2)

    diff = (out1[0].float() - out2[0].float()).abs()
    print(f"  Linear B=1 vs B=2: max={diff.max():.6f}, mean={diff.mean():.8f}")
    if diff.max() < 1e-5:
        print("  => Linear is batch-independent.\n")
    else:
        print("  => Linear has batch-dependent behavior!\n")
    return diff.max().item()


def test_embedding(device, model_path):
    """Test embedding layer with different batch sizes."""
    print("=" * 70)
    print("Test 4: Embedding, B=1 vs B=2")
    print("=" * 70)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa").to(device).eval()

    embed = model.model.embed_tokens

    ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    out1 = embed(ids)

    ids2 = ids.expand(2, -1).contiguous()
    out2 = embed(ids2)

    diff = (out1[0].float() - out2[0].float()).abs()
    print(f"  Embedding B=1 vs B=2: max={diff.max():.6f}, mean={diff.mean():.8f}\n")

    del model
    torch.npu.empty_cache()
    return diff.max().item()


def test_single_layer(device, model_path):
    """Test a single transformer layer with different batch sizes."""
    print("=" * 70)
    print("Test 5: Single transformer layer, B=1 vs B=2")
    print("=" * 70)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa").to(device).eval()

    layer0 = model.model.layers[0]
    H = model.config.hidden_size

    torch.manual_seed(42)
    x1 = torch.randn(1, 128, H, dtype=torch.bfloat16, device=device)
    pos1 = torch.arange(128, device=device).unsqueeze(0)
    mask1 = torch.ones(1, 128, device=device, dtype=torch.long)

    # Need position embeddings
    pos_emb = model.model.rotary_emb(x1, pos1)

    with torch.no_grad():
        out1 = layer0(x1, position_ids=pos1, position_embeddings=pos_emb)[0]

    x2 = x1.expand(2, -1, -1).contiguous()
    pos2 = pos1.expand(2, -1).contiguous()
    pos_emb2 = (pos_emb[0].expand(2, -1, -1).contiguous(),
                pos_emb[1].expand(2, -1, -1).contiguous())

    with torch.no_grad():
        out2 = layer0(x2, position_ids=pos2, position_embeddings=pos_emb2)[0]

    diff = (out1[0].float() - out2[0].float()).abs()
    print(f"  Layer 0 B=1 vs B=2: max={diff.max():.6f}, mean={diff.mean():.8f}")
    if diff.max() < 1e-5:
        print("  => Single layer is batch-independent.\n")
    else:
        print("  => Single layer has batch-dependent behavior!\n")

    del model
    torch.npu.empty_cache()
    return diff.max().item()


def test_full_model(device, model_path):
    """Test full model B=1 vs B=2."""
    print("=" * 70)
    print("Test 6: Full model, B=1 vs B=2 (no padding)")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa").to(device).eval()

    ids = tokenizer.encode("def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)\n" * 5,
                           add_special_tokens=False)[:128]

    s1 = torch.tensor([ids], device=device)
    m1 = torch.ones_like(s1)
    p1 = torch.arange(len(ids), device=device).unsqueeze(0)
    with torch.no_grad():
        o1 = model(s1, attention_mask=m1, position_ids=p1).logits

    s2 = torch.tensor([ids, ids], device=device)
    m2 = torch.ones_like(s2)
    p2 = torch.arange(len(ids), device=device).unsqueeze(0).expand(2, -1).contiguous()
    with torch.no_grad():
        o2 = model(s2, attention_mask=m2, position_ids=p2).logits

    diff = (o1[0].float() - o2[0].float()).abs()
    print(f"  Full model B=1 vs B=2: max={diff.max():.6f}, mean={diff.mean():.8f}")

    # Logprob impact
    labels = torch.roll(s1, -1, 1)
    lp1 = F.log_softmax(o1.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0, :-1]
    lp2 = F.log_softmax(o2[0:1].float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0, :-1]
    lp_diff = (lp1 - lp2).abs()
    print(f"  Logprob diff: max={lp_diff.max():.6f}, mean={lp_diff.mean():.6f}")
    print(f"  Tokens > GSPO threshold (0.0003): {(lp_diff > 0.0003).sum().item()}/{lp_diff.numel()} "
          f"({(lp_diff > 0.0003).float().mean()*100:.1f}%)")

    # Test B=1 vs B=1 (determinism check)
    with torch.no_grad():
        o1b = model(s1, attention_mask=m1, position_ids=p1).logits
    det_diff = (o1[0].float() - o1b[0].float()).abs()
    print(f"\n  Determinism check (B=1 twice): max={det_diff.max():.6f}")

    del model
    torch.npu.empty_cache()
    return diff.max().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print()

    results = {}
    results["attention_kernel"] = test_raw_attention_kernel(args.device)
    results["rmsnorm"] = test_rmsnorm(args.device)
    results["linear"] = test_linear(args.device)
    results["embedding"] = test_embedding(args.device, args.model)
    results["single_layer"] = test_single_layer(args.device, args.model)
    results["full_model"] = test_full_model(args.device, args.model)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, val in results.items():
        status = "OK" if val < 1e-5 else "BUG"
        print(f"  {name:20s}: max_diff={val:.6f}  [{status}]")

    buggy = [k for k, v in results.items() if v > 1e-5]
    if buggy:
        print(f"\n  Batch-dependent components: {', '.join(buggy)}")
        print(f"  These cause NPU training to diverge from H200.")
        print(f"\n  Workaround: process sequences one at a time (B=1) in forward pass,")
        print(f"  then concatenate results. This matches H200 behavior.")
    else:
        print(f"\n  All components are batch-independent.")
        print(f"  The bug may be in how transformers constructs the causal mask.")


if __name__ == "__main__":
    main()
