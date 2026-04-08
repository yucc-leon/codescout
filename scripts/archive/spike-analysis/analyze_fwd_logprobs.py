#!/usr/bin/env python3
"""分析 fwd_logprobs_values_reward 耗时和 glen 的关系，
以及它是否影响后续的 policy_train。"""
import re
import statistics

LOG = "/sharedata/liyuchen/workspace/logs/0401_1643_Qwen3-4B-Instruct-2507-npu-8x8.log"
PID = "skyrl_entrypoint pid=2200775"

# Extract all timed operations per step
steps = []
current_step = {}

with open(LOG) as f:
    for line in f:
        if PID not in line:
            continue
        # fwd_logprobs_values_reward
        m = re.search(r"Finished: 'fwd_logprobs_values_reward', time cost: ([0-9.]+)s", line)
        if m:
            current_step["fwd_logprobs"] = float(m.group(1))
            continue
        # policy_train
        m = re.search(r"Finished: 'policy_train', time cost: ([0-9.]+)s", line)
        if m:
            current_step["policy_train"] = float(m.group(1))
            if "fwd_logprobs" in current_step:
                steps.append(dict(current_step))
            current_step = {}

# Get glen data
glen_data = []
with open(LOG) as f:
    for line in f:
        if "Policy Train epoch" not in line or "100%" not in line:
            continue
        m = re.search(r"glen=([0-9.e+]+)", line)
        if m:
            glen_data.append(float(m.group(1)))

# Deduplicate glen (pairs)
glen_unique = glen_data[::2]

n = min(len(steps), len(glen_unique))
print(f"Steps: {len(steps)}, Glen: {len(glen_unique)}, Paired: {n}")

print(f"\n{'step':>4}  {'glen':>8}  {'fwd_lp(s)':>10}  {'policy(s)':>10}  {'total(s)':>10}  {'fwd_lp%':>8}")
print("-" * 60)

for i in range(min(n, 30)):
    glen = glen_unique[i]
    fwd = steps[i]["fwd_logprobs"]
    pol = steps[i]["policy_train"]
    total = fwd + pol
    pct = fwd / total * 100
    print(f"{i:>4}  {glen:>8.0f}  {fwd:>10.1f}  {pol:>10.1f}  {total:>10.1f}  {pct:>7.1f}%")

# Correlation: glen vs policy_train
glens = [glen_unique[i] for i in range(n)]
pol_times = [steps[i]["policy_train"] for i in range(n)]
fwd_times = [steps[i]["fwd_logprobs"] for i in range(n)]

def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = sum((x-mx)**2 for x in xs)**0.5
    sy = sum((y-my)**2 for y in ys)**0.5
    if sx == 0 or sy == 0: return 0
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / (sx*sy)

print(f"\nr(glen, policy_train) = {pearson(glens, pol_times):.4f}")
print(f"r(glen, fwd_logprobs) = {pearson(glens, fwd_times):.4f}")
print(f"r(fwd_logprobs, policy_train) = {pearson(fwd_times, pol_times):.4f}")

# Key question: does fwd_logprobs time predict policy_train spike?
# If fwd_logprobs leaves memory pressure, policy_train would be slow
# regardless of glen
print(f"\n=== 控制 glen 后看 fwd_logprobs 对 policy_train 的影响 ===")
# Group by glen bucket, see if fwd_logprobs variance explains policy_train variance
buckets = {}
for i in range(n):
    b = int(glen_unique[i] // 5000) * 5000
    buckets.setdefault(b, []).append((fwd_times[i], pol_times[i]))

for b in sorted(buckets.keys()):
    items = buckets[b]
    if len(items) >= 3:
        fwd_vals = [x[0] for x in items]
        pol_vals = [x[1] for x in items]
        r = pearson(fwd_vals, pol_vals) if len(items) >= 3 else 0
        print(f"  glen {b}-{b+5000}: n={len(items)}, r(fwd,pol)={r:.3f}")
