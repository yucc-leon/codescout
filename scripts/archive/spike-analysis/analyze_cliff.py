#!/usr/bin/env python3
"""精确定位 SDPA 性能悬崖。"""
import re
import statistics

LOG = "/sharedata/liyuchen/workspace/logs/0401_1643_Qwen3-4B-Instruct-2507-npu-8x8.log"

data = []
with open(LOG) as f:
    for line in f:
        if "Policy Train epoch" not in line or "100%" not in line:
            continue
        m_time = re.search(r"\[(\d+):(\d+)<", line)
        m_glen = re.search(r"glen=([0-9.e+]+)", line)
        if m_time and m_glen:
            time_s = int(m_time.group(1)) * 60 + int(m_time.group(2))
            glen = float(m_glen.group(1))
            data.append((glen, time_s))

unique = data[::2]

# 更细的分桶（每 2000 token）
print("=== 细粒度分桶 (2000 token) ===")
print(f"{'glen 区间':>20}  {'n':>3}  {'mean(s)':>8}  {'median(s)':>8}  {'per_iter(s)':>12}  {'ms/token':>10}")
print("-" * 75)
buckets = {}
for glen, time in unique:
    b = int(glen // 2000) * 2000
    buckets.setdefault(b, []).append((glen, time))

prev_per_token = None
for b in sorted(buckets.keys()):
    items = buckets[b]
    times = [t for _, t in items]
    glens = [g for g, _ in items]
    mean_t = statistics.mean(times)
    median_t = statistics.median(times)
    mean_glen = statistics.mean(glens)
    # 16 micro-batches per step
    per_iter = mean_t / 16
    ms_per_token = mean_t / mean_glen * 1000
    
    jump = ""
    if prev_per_token:
        ratio = ms_per_token / prev_per_token
        if ratio > 1.3:
            jump = f"  ← JUMP {ratio:.2f}x"
    
    print(f"  glen {b:>6}-{b+2000:>6}  {len(items):>3}  {mean_t:>8.0f}  {median_t:>8.0f}  {per_iter:>12.1f}  {ms_per_token:>10.2f}{jump}")
    prev_per_token = ms_per_token

# 看 glen > 34000 的数据点
print(f"\n=== glen > 34000 的详细数据 ===")
print(f"{'glen':>8}  {'time(s)':>8}  {'per_iter(s)':>12}  {'ms/token':>10}")
high_data = [(g, t) for g, t in unique if g > 34000]
high_data.sort()
for glen, time in high_data:
    per_iter = time / 16
    ms_per_token = time / glen * 1000
    print(f"{glen:>8.0f}  {time:>8.0f}  {per_iter:>12.1f}  {ms_per_token:>10.2f}")

# 对比 glen < 34000 的数据
print(f"\n=== glen < 34000 的统计 ===")
low_data = [(g, t) for g, t in unique if g < 34000]
low_ms = [t / g * 1000 for g, t in low_data]
print(f"n={len(low_data)}, ms/token: mean={statistics.mean(low_ms):.2f}, stdev={statistics.stdev(low_ms):.2f}")

print(f"\n=== glen >= 34000 的统计 ===")
high_data2 = [(g, t) for g, t in unique if g >= 34000]
high_ms = [t / g * 1000 for g, t in high_data2]
print(f"n={len(high_data2)}, ms/token: mean={statistics.mean(high_ms):.2f}, stdev={statistics.stdev(high_ms):.2f}")

print(f"\n性能退化倍数: {statistics.mean(high_ms) / statistics.mean(low_ms):.2f}x")

# 看 max_seq_len 配置
print(f"\n=== 配置信息 ===")
print(f"max_seq_len: 49152 (from log)")
print(f"micro_train_batch_size_per_gpu: 1")
print(f"16 micro-batches per step (from tqdm 16/16)")
print(f"8x8 = 64 NPUs, 64 sequences per batch")
print(f"glen = prompt_len + max_response_len (padded)")
