#!/usr/bin/env python3
"""
对比训练 log 中的 per-iteration 时间 vs 独立 SDPA benchmark。
看 per-iteration 时间是否和 SDPA 时间成比例，还是有额外的非线性因素。
"""
import re

LOG = "/sharedata/liyuchen/workspace/logs/0401_1643_Qwen3-4B-Instruct-2507-npu-8x8.log"

# 独立 SDPA benchmark 结果 (S -> fwd+bwd ms)
SDPA_BENCH = {
    8192: 12.4,
    12288: 26.2,
    16384: 45.4,
    20480: 70.3,
    24576: 100.3,
    28672: 135.8,
    30720: 155.0,
    31744: 165.6,
    32768: 176.8,
    33792: 187.4,
    34816: 199.0,
    35840: 209.7,
    36864: 222.2,
    37888: 234.2,
    40960: 273.5,
}

def interpolate_sdpa(glen):
    """线性插值 SDPA benchmark 结果。"""
    keys = sorted(SDPA_BENCH.keys())
    if glen <= keys[0]:
        return SDPA_BENCH[keys[0]] * glen / keys[0]
    if glen >= keys[-1]:
        return SDPA_BENCH[keys[-1]] * glen / keys[-1]
    for i in range(len(keys) - 1):
        if keys[i] <= glen <= keys[i+1]:
            frac = (glen - keys[i]) / (keys[i+1] - keys[i])
            return SDPA_BENCH[keys[i]] + frac * (SDPA_BENCH[keys[i+1]] - SDPA_BENCH[keys[i]])
    return 0

def parse_time(time_str):
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])

# Extract data
data = []
with open(LOG) as f:
    for line in f:
        if "Policy Train epoch" not in line or "100%" not in line:
            continue
        m_time = re.search(r"\[(\d+:\d+)<", line)
        m_glen = re.search(r"glen=([0-9.e+]+)", line)
        m_per_it = re.search(r"(\d+\.\d+)s/it", line)
        if m_time and m_glen and m_per_it:
            glen = float(m_glen.group(1))
            total_s = parse_time(m_time.group(1))
            per_it = float(m_per_it.group(1))
            data.append((glen, total_s, per_it))

# Deduplicate (pairs)
unique = data[::2]

print(f"{'glen':>8}  {'per_it(s)':>10}  {'sdpa_est(ms)':>12}  {'36L_sdpa(s)':>12}  {'non_sdpa(s)':>12}  {'sdpa_frac':>10}")
print("-" * 80)

for glen, total_s, per_it in sorted(unique, key=lambda x: x[0]):
    sdpa_ms = interpolate_sdpa(glen)
    # 36 layers of attention per forward+backward pass
    sdpa_36L_s = sdpa_ms * 36 / 1000
    non_sdpa_s = per_it - sdpa_36L_s
    sdpa_frac = sdpa_36L_s / per_it if per_it > 0 else 0
    print(f"{glen:>8.0f}  {per_it:>10.1f}  {sdpa_ms:>12.1f}  {sdpa_36L_s:>12.1f}  {non_sdpa_s:>12.1f}  {sdpa_frac:>10.1%}")

# 分桶分析
import statistics
print(f"\n=== 分桶分析 ===")
print(f"{'glen 区间':>20}  {'n':>3}  {'mean per_it':>12}  {'mean sdpa_frac':>14}  {'mean non_sdpa':>14}")
print("-" * 75)
buckets = {}
for glen, total_s, per_it in unique:
    sdpa_ms = interpolate_sdpa(glen)
    sdpa_36L_s = sdpa_ms * 36 / 1000
    non_sdpa_s = per_it - sdpa_36L_s
    sdpa_frac = sdpa_36L_s / per_it if per_it > 0 else 0
    b = int(glen // 5000) * 5000
    buckets.setdefault(b, []).append((per_it, sdpa_frac, non_sdpa_s))

for b in sorted(buckets.keys()):
    items = buckets[b]
    mean_per_it = statistics.mean([x[0] for x in items])
    mean_frac = statistics.mean([x[1] for x in items])
    mean_non_sdpa = statistics.mean([x[2] for x in items])
    print(f"  glen {b:>6}-{b+5000:>6}  {len(items):>3}  {mean_per_it:>12.1f}s  {mean_frac:>14.1%}  {mean_non_sdpa:>14.1f}s")
