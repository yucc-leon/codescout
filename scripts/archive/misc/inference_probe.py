"""
Inference probe: test vLLM output on fixed prompts.
Run this on BOTH NPU and H200 with the same base model, compare results.

Usage:
  python codescout/scripts/inference_probe.py --port 8100
"""

import argparse, json, requests, hashlib

PROMPTS = [
    # Prompt 1: Simple code search
    {
        "messages": [
            {"role": "system", "content": "You are a code search agent. Use tools to find relevant files."},
            {"role": "user", "content": "Find the file that handles user authentication in a Django project. The project root is /tmp/testbed/django__django-16527."},
        ],
        "id": "probe_1_auth"
    },
    # Prompt 2: More specific
    {
        "messages": [
            {"role": "system", "content": "You are a code search agent. Use tools to find relevant files."},
            {"role": "user", "content": "Find the module responsible for database migrations in /tmp/testbed/django__django-16527."},
        ],
        "id": "probe_2_migrations"
    },
    # Prompt 3: Different domain
    {
        "messages": [
            {"role": "system", "content": "You are a code search agent. Use tools to find relevant files."},
            {"role": "user", "content": "Locate the file that implements the HTTP request handling in /tmp/testbed/requests__requests-5414."},
        ],
        "id": "probe_3_requests"
    },
]

def run_probe(base_url, model_name):
    results = []
    for p in PROMPTS:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": p["messages"],
                "max_tokens": 200,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 5,
            },
            timeout=120,
        )
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        
        # Extract logprobs
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = []
        if logprobs_data and logprobs_data.get("content"):
            for entry in logprobs_data["content"][:20]:  # first 20 tokens
                token_logprobs.append({
                    "token": entry["token"],
                    "logprob": entry["logprob"],
                    "top5": [
                        {"token": t["token"], "logprob": t["logprob"]}
                        for t in entry.get("top_logprobs", [])[:5]
                    ]
                })
        
        result = {
            "id": p["id"],
            "content_first_200": content[:200],
            "content_hash": hashlib.md5(content.encode()).hexdigest(),
            "num_tokens": len(logprobs_data.get("content", [])) if logprobs_data else 0,
            "first_20_tokens": token_logprobs,
            "finish_reason": choice.get("finish_reason"),
        }
        results.append(result)
        print(f"\n{'='*60}")
        print(f"Prompt: {p['id']}")
        print(f"Output ({result['num_tokens']} tokens, hash={result['content_hash']}):")
        print(f"  {content[:200]}")
        print(f"First 10 tokens with logprobs:")
        for t in token_logprobs[:10]:
            top1_alt = t["top5"][1]["token"] if len(t["top5"]) > 1 else "N/A"
            top1_alt_lp = t["top5"][1]["logprob"] if len(t["top5"]) > 1 else 0
            gap = t["logprob"] - top1_alt_lp if len(t["top5"]) > 1 else float('inf')
            print(f"  '{t['token']}' lp={t['logprob']:.4f}  (2nd: '{top1_alt}' lp={top1_alt_lp:.4f}, gap={gap:.4f})")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    # Get model name
    models = requests.get(f"{base_url}/v1/models", timeout=10).json()
    model_name = models["data"][0]["id"]
    print(f"Model: {model_name}")
    
    results = run_probe(base_url, model_name)
    
    output_file = args.output or f"/tmp/inference_probe_results.json"
    with open(output_file, "w") as f:
        json.dump({"model": model_name, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
