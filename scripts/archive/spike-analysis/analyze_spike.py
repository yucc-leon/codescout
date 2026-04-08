#!/usr/bin/env python3
"""分析 policy_train 耗时 spike 与序列长度的关联性。

从训练 log 中提取 per-step 的 (avg_response_length, policy_train_time) 配对，
计算相关系数，输出分析结果。
"""
import re
import sys
import statistics

LOG = "/sharedata/liyuchen/workspace/logs/0401_1643_Qwen3-4B-Instruct-2507-npu-8x8.log"
PID_FILTER = "skyrl_entrypoint pid=2200775"

def extract_pairs(logfile: str) -> list[tuple[float, float]]:
    """从 log 中提取 (avg_response_length, policy_train_time) 配对。"""
    pairs = []
    pending_length = None

    with open(logfile) as f:
        for line in f:
            if PID_FILTER not in line:
                continue
            # avg_response_length line
            m = re.search(r"avg_response_length:\s*([0-9.]+)", line)
            if m:
                pending_length = float(m.group(1))
                continue
            # policy_train time line
            m = re.search(r"Finished: 'policy_train', time cost: ([0-9.]+)s", line)
            if m and pending_length is not None:
                pairs.append((pending_length, float(m.group(1))))
                pending_length = None

    return pairs

def pearson_r(xs, ys):
    """手算 Pearson 相关系数，避免依赖 numpy/scipy。"""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)

def main():
    pairs = extract_pairs(LOG)
    if not pairs:
        print("未找到配对数据")
        sys.exit(1)

    lengths = [p[0] for p in pairs]
    times = [p[1] for p in pairs]

    print(f"共 {len(pairs)} 个 step\n")

    # 基本统计
    print("=== policy_train 耗时统计 ===")
    print(f"  min:    {min(times):>8.1f}s")
    print(f"  max:    {max(times):>8.1f}s")
    print(f"  mean:   {statistics.mean(times):>8.1f}s")
    print(f"  median: {statistics.median(times):>8.1f}s")
    print(f"  stdev:  {statistics.stdev(times):>8.1f}s")

    print(f"\n=== avg_response_length 统计 ===")
    print(f"  min:    {min(lengths):>8.1f}")
    print(f"  max:    {max(lengths):>8.1f}")
    print(f"  mean:   {statistics.mean(lengths):>8.1f}")
    print(f"  median: {statistics.median(lengths):>8.1f}")

    # 相关性
    r = pearson_r(lengths, times)
    print(f"\n=== 相关性分析 ===")
    print(f"  Pearson r(avg_response_length, policy_train_time) = {r:.4f}")
    if abs(r) > 0.7:
        print("  → 强相关：序列长度是 spike 的主要因素")
    elif abs(r) > 0.4:
        print("  → 中等相关：序列长度有影响，但不是唯一因素")
    else:
        print("  → 弱相关：序列长度不是主要因素，需要排查其他原因")

    # 分桶分析：按序列长度分桶，看每个桶的平均耗时
    print(f"\n=== 按序列长度分桶 ===")
    print(f"{'长度区间':>16}  {'步数':>4}  {'平均耗时':>8}  {'最大耗时':>8}  {'最小耗时':>8}")
    print("-" * 60)
    buckets = {}
    for length, time in pairs:
        bucket = int(length // 2000) * 2000
        key = f"{bucket}-{bucket+2000}"
        buckets.setdefault(key, []).append(time)

    for key in sorted(buckets.keys(), key=lambda x: int(x.split("-")[0])):
        ts = buckets[key]
        print(f"{key:>16}  {len(ts):>4}  {statistics.mean(ts):>8.1f}s  {max(ts):>8.1f}s  {min(ts):>8.1f}s")

    # Top 20 最慢 step
    print(f"\n=== Top 20 最慢 step ===")
    print(f"{'#':>3}  {'耗时':>10}  {'avg_resp_len':>12}")
    sorted_pairs = sorted(pairs, key=lambda p: p[1], reverse=True)
    for i, (length, time) in enumerate(sorted_pairs[:20]):
        print(f"{i+1:>3}  {time:>10.1f}s  {length:>12.1f}")

    # Top 20 最快 step
    print(f"\n=== Top 20 最快 step ===")
    print(f"{'#':>3}  {'耗时':>10}  {'avg_resp_len':>12}")
    for i, (length, time) in enumerate(sorted_pairs[-20:]):
        print(f"{i+1:>3}  {time:>10.1f}s  {length:>12.1f}")

    # 检查是否有 "同样长度但耗时差异大" 的情况
    print(f"\n=== 同长度区间内的耗时方差 ===")
    for key in sorted(buckets.keys(), key=lambda x: int(x.split("-")[0])):
        ts = buckets[key]
        if len(ts) >= 3:
            cv = statistics.stdev(ts) / statistics.mean(ts) * 100
            print(f"  {key}: CV={cv:.1f}% (n={len(ts)})")

def extract_glen_pairs(logfile: str) -> list[tuple[float, float]]:
    """从 tqdm 输出提取 (glen, policy_train_time) 配对。
    
    glen 是 padding 后的 max response length，反映 batch 内最长序列。
    """
    import re
    glen_values = []
    # 从 100% 完成的 tqdm 行提取 glen
    with open(logfile) as f:
        for line in f:
            if "Policy Train epoch" not in line or "100%" not in line:
                continue
            m = re.search(r"glen=([0-9.e+]+)", line)
            if m:
                glen_str = m.group(1)
                glen_values.append(float(glen_str))

    # 从 entrypoint 提取 policy_train times
    times = []
    with open(logfile) as f:
        for line in f:
            if PID_FILTER not in line:
                continue
            m = re.search(r"Finished: 'policy_train', time cost: ([0-9.]+)s", line)
            if m:
                times.append(float(m.group(1)))

    # 配对
    n = min(len(glen_values), len(times))
    if n == 0:
        return []
    return list(zip(glen_values[:n], times[:n]))


def analyze_glen_correlation():
    """分析 glen (max padded response length) 与 policy_train 耗时的关联。"""
    pairs = extract_glen_pairs(LOG)
    if not pairs:
        print("\n未找到 glen 数据")
        return

    glens = [p[0] for p in pairs]
    times = [p[1] for p in pairs]

    r = pearson_r(glens, times)
    print(f"\n{'='*60}")
    print(f"=== glen (max padded response length) 关联分析 ===")
    print(f"{'='*60}")
    print(f"共 {len(pairs)} 个配对")
    print(f"Pearson r(glen, policy_train_time) = {r:.4f}")
    if abs(r) > 0.7:
        print("→ 强相关：batch 内最长序列长度是 spike 的主要因素")
    elif abs(r) > 0.4:
        print("→ 中等相关")
    else:
        print("→ 弱相关")

    # 分桶
    print(f"\n{'glen 区间':>20}  {'步数':>4}  {'平均耗时':>10}  {'最大耗时':>10}  {'最小耗时':>10}")
    print("-" * 70)
    buckets = {}
    for glen, time in pairs:
        bucket = int(glen // 5000) * 5000
        key = f"{bucket}-{bucket+5000}"
        buckets.setdefault(key, []).append(time)

    for key in sorted(buckets.keys(), key=lambda x: int(x.split("-")[0])):
        ts = buckets[key]
        print(f"{key:>20}  {len(ts):>4}  {statistics.mean(ts):>10.1f}s  {max(ts):>10.1f}s  {min(ts):>10.1f}s")

    # 散点数据（方便后续画图）
    print(f"\n=== glen vs time 散点数据 ===")
    print(f"{'glen':>8}  {'time(s)':>10}")
    for glen, time in sorted(pairs, key=lambda p: p[0]):
        print(f"{glen:>8.0f}  {time:>10.1f}")


def analyze_temporal_pattern():
    """分析 spike 的时间模式：是否有周期性？"""
    pairs = extract_pairs(LOG)
    times = [p[1] for p in pairs]

    print("\n=== 时间序列模式 ===")
    print(f"{'step':>4}  {'耗时':>10}  {'avg_len':>10}  {'spike?':>6}")
    print("-" * 40)
    threshold = statistics.mean(times) + statistics.stdev(times)  # ~900s
    spike_steps = []
    for i, (length, time) in enumerate(pairs):
        is_spike = "***" if time > threshold else ""
        if time > threshold:
            spike_steps.append(i)
        print(f"{i:>4}  {time:>10.1f}s  {length:>10.1f}  {is_spike}")

    print(f"\nSpike 阈值: {threshold:.0f}s")
    print(f"Spike 步数: {len(spike_steps)} / {len(pairs)}")

    if len(spike_steps) > 1:
        gaps = [spike_steps[i+1] - spike_steps[i] for i in range(len(spike_steps)-1)]
        print(f"Spike 间隔: {gaps}")
        print(f"平均间隔: {statistics.mean(gaps):.1f} steps")
        if len(gaps) >= 2:
            print(f"间隔 stdev: {statistics.stdev(gaps):.1f}")

    # 看连续 spike 的情况
    print(f"\n=== 连续 spike 检测 ===")
    streak = 0
    max_streak = 0
    for i, (_, time) in enumerate(pairs):
        if time > threshold:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            if streak > 1:
                print(f"  连续 spike: step {i-streak} ~ {i-1} ({streak} 步)")
            streak = 0
    if streak > 1:
        print(f"  连续 spike: step {len(pairs)-streak} ~ {len(pairs)-1} ({streak} 步)")
    print(f"  最长连续 spike: {max_streak} 步")


if __name__ == "__main__":
    main()
    analyze_glen_correlation()
    analyze_temporal_pattern()
