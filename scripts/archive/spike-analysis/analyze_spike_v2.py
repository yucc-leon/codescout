#!/usr/bin/env python3
"""分析 policy_train spike v2: 对比 rank_0 worker 自身耗时 vs entrypoint 总耗时。

如果 rank_0 耗时 << entrypoint 耗时，说明 spike 来自其他 DP worker（straggler）。
如果 rank_0 耗时 ≈ entrypoint 耗时，说明 spike 来自 rank_0 自身的计算。
"""
import re
import statistics

LOG = "/sharedata/liyuchen/workspace/logs/0401_1643_Qwen3-4B-Instruct-2507-npu-8x8.log"
PID_FILTER = "skyrl_entrypoint pid=2200775"

def parse_tqdm_time(time_str: str) -> float:
    """Parse MM:SS format to seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])

def main():
    # Extract rank_0 worker tqdm data: (glen, worker_time_seconds)
    worker_data = []
    with open(LOG) as f:
        for line in f:
            if "Policy Train epoch" not in line or "100%" not in line:
                continue
            m_time = re.search(r'\[(\d+:\d+)<', line)
            m_glen = re.search(r'glen=([0-9.e+]+)', line)
            if m_time and m_glen:
                worker_time = parse_tqdm_time(m_time.group(1))
                glen = float(m_glen.group(1))
                worker_data.append((glen, worker_time))

    # Extract entrypoint policy_train times
    entrypoint_times = []
    with open(LOG) as f:
        for line in f:
            if PID_FILTER not in line:
                continue
            m = re.search(r"Finished: 'policy_train', time cost: ([0-9.]+)s", line)
            if m:
                entrypoint_times.append(float(m.group(1)))

    n = min(len(worker_data), len(entrypoint_times))
    print(f"配对数: {n}")
    print(f"Worker (rank_0) 数据: {len(worker_data)}")
    print(f"Entrypoint 数据: {len(entrypoint_times)}")

    # Compare
    print(f"\n{'step':>4}  {'glen':>8}  {'rank0(s)':>10}  {'total(s)':>10}  {'diff(s)':>10}  {'ratio':>6}  {'straggler?':>10}")
    print("-" * 75)

    diffs = []
    straggler_count = 0
    for i in range(n):
        glen, worker_time = worker_data[i]
        total_time = entrypoint_times[i]
        diff = total_time - worker_time
        ratio = total_time / worker_time if worker_time > 0 else float('inf')
        is_straggler = "YES" if diff > 100 else ""
        if diff > 100:
            straggler_count += 1
        diffs.append(diff)
        print(f"{i:>4}  {glen:>8.0f}  {worker_time:>10.0f}  {total_time:>10.1f}  {diff:>10.1f}  {ratio:>6.2f}  {is_straggler:>10}")

    print(f"\n=== 总结 ===")
    print(f"Straggler 步数 (diff > 100s): {straggler_count} / {n}")
    print(f"Diff 统计: mean={statistics.mean(diffs):.1f}s, median={statistics.median(diffs):.1f}s, max={max(diffs):.1f}s, min={min(diffs):.1f}s")

    # 看 rank_0 自身的 glen vs time 相关性
    worker_glens = [d[0] for d in worker_data[:n]]
    worker_times = [d[1] for d in worker_data[:n]]

    # Pearson r
    mx = statistics.mean(worker_glens)
    my = statistics.mean(worker_times)
    sx = sum((x - mx)**2 for x in worker_glens) ** 0.5
    sy = sum((y - my)**2 for y in worker_times) ** 0.5
    if sx > 0 and sy > 0:
        r = sum((x - mx) * (y - my) for x, y in zip(worker_glens, worker_times)) / (sx * sy)
        print(f"\nPearson r(glen, rank0_time) = {r:.4f}")
    
    r2_glens = [d[0] for d in worker_data[:n]]
    r2_times = entrypoint_times[:n]
    mx2 = statistics.mean(r2_glens)
    my2 = statistics.mean(r2_times)
    sx2 = sum((x - mx2)**2 for x in r2_glens) ** 0.5
    sy2 = sum((y - my2)**2 for y in r2_times) ** 0.5
    if sx2 > 0 and sy2 > 0:
        r2 = sum((x - mx2) * (y - my2) for x, y in zip(r2_glens, r2_times)) / (sx2 * sy2)
        print(f"Pearson r(glen, total_time) = {r2:.4f}")

    # 分析 spike 步的 diff 分布
    print(f"\n=== Spike 步 (total > 900s) 的 rank_0 vs total ===")
    for i in range(n):
        glen, worker_time = worker_data[i]
        total_time = entrypoint_times[i]
        if total_time > 900:
            diff = total_time - worker_time
            print(f"  step {i}: glen={glen:.0f}, rank0={worker_time:.0f}s, total={total_time:.1f}s, diff={diff:.1f}s")

if __name__ == "__main__":
    main()
