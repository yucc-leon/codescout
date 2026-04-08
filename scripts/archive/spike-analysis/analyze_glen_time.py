#!/usr/bin/env python3
"""分析 glen vs time 的关系：线性？二次？有性能悬崖？"""
import re
import statistics

LOG = "/sharedata/liyuchen/workspace/logs/0401_1643_Qwen3-4B-Instruct-2507-npu-8x8.log"

# Extract unique (glen, time) pairs
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

# Deduplicate (take every other one since they're pairs)
unique = data[::2]

print(f"共 {len(unique)} 个 step")
print(f"glen 范围: {min(d[0] for d in unique):.0f} ~ {max(d[0] for d in unique):.0f}")
print(f"time 范围: {min(d[1] for d in unique):.0f}s ~ {max(d[1] for d in unique):.0f}s")

# 看 time/glen 的比率是否稳定（线性关系）
print(f"\n=== time/glen 比率 (ms/token) ===")
ratios = [d[1] / d[0] * 1000 for d in unique]
print(f"mean: {statistics.mean(ratios):.2f} ms/token")
print(f"stdev: {statistics.stdev(ratios):.2f} ms/token")
print(f"CV: {statistics.stdev(ratios)/statistics.mean(ratios)*100:.1f}%")

# 看 time/glen^2 的比率（二次关系）
ratios2 = [d[1] / (d[0]**2) * 1e9 for d in unique]
print(f"\n=== time/glen^2 比率 (ns/token^2) ===")
print(f"mean: {statistics.mean(ratios2):.4f}")
print(f"stdev: {statistics.stdev(ratios2):.4f}")
print(f"CV: {statistics.stdev(ratios2)/statistics.mean(ratios2)*100:.1f}%")

# 分桶看增长率
print(f"\n=== 分桶增长率 ===")
buckets = {}
for glen, time in unique:
    b = int(glen // 5000) * 5000
    buckets.setdefault(b, []).append(time)

prev_mean = None
for b in sorted(buckets.keys()):
    ts = buckets[b]
    mean_t = statistics.mean(ts)
    ratio_str = ""
    if prev_mean:
        ratio_str = f"  增长 {mean_t/prev_mean:.2f}x"
    print(f"  glen {b:>6}-{b+5000:>6}: n={len(ts):>3}, mean={mean_t:>6.0f}s, median={statistics.median(ts):>6.0f}s{ratio_str}")
    prev_mean = mean_t

# 简单线性回归
n = len(unique)
xs = [d[0] for d in unique]
ys = [d[1] for d in unique]
mx = statistics.mean(xs)
my = statistics.mean(ys)
ss_xx = sum((x - mx)**2 for x in xs)
ss_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
slope = ss_xy / ss_xx
intercept = my - slope * mx

print(f"\n=== 线性回归: time = {slope:.6f} * glen + {intercept:.1f} ===")
print(f"即: 每增加 1000 token, 耗时增加 {slope * 1000:.1f}s")

# R^2
ss_res = sum((y - (slope * x + intercept))**2 for x, y in zip(xs, ys))
ss_tot = sum((y - my)**2 for y in ys)
r_squared = 1 - ss_res / ss_tot
print(f"R^2 = {r_squared:.4f}")

# 残差分析
residuals = [(x, y, y - (slope * x + intercept)) for x, y in zip(xs, ys)]
print(f"\n=== 残差最大的 10 个点 ===")
residuals.sort(key=lambda r: abs(r[2]), reverse=True)
for glen, time, res in residuals[:10]:
    predicted = slope * glen + intercept
    print(f"  glen={glen:.0f}, actual={time:.0f}s, predicted={predicted:.0f}s, residual={res:.0f}s")
