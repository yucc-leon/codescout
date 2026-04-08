#!/usr/bin/env python3
"""分析 policy_train spike v3: 验证 DP 负载不均衡假设。

8x8 配置 = 8 nodes × 8 NPUs = 64 NPUs total。
如果 FSDP DP=2，则有 2 个 DP group，每个 group 32 NPUs。
每个 DP group 处理不同的 batch，glen 不同。
policy_train 时间 = max(group_0_time, group_1_time)。

rank_0 的 glen 成对出现（连续两个 step 共享同一个 glen），
说明每个 "training step" 实际上包含 2 个 mini-batch（update_epochs=1, 
但 batch 被分成 2 个 mini-batch 给 2 个 DP group）。

验证：如果 spike 是因为 DP 负载不均衡，那么：
- 当 rank_0 的 glen 小时，total 应该大（另一个 group 的 glen 大）
- 当 rank_0 的 glen 大时，total 应该小（另一个 group 的 glen 小）
"""
import re
import statistics

LOG = "/sharedata/liyuchen/workspace/logs/0401_1643_Qwen3-4B-Instruct-2507-npu-8x8.log"
PID_FILTER = "skyrl_entrypoint pid=2200775"

def parse_tqdm_time(time_str: str) -> float:
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])

def main():
    # Extract rank_0 tqdm data
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

    # glen 成对出现，每对对应一个 entrypoint step
    # 但实际上 268 个 worker_data 对应 134 个 entrypoint_times
    # 这意味着每个 entrypoint step 有 2 个 tqdm 输出（2 个 mini-batch？）
    # 或者 rank_0 处理了 2 个 DP group 的数据？
    
    # 实际上更可能是：rank_0 的 tqdm 输出是连续的，
    # 每个 entrypoint step 对应 rank_0 的 2 个 epoch 或 2 个 mini-batch
    
    print(f"Worker data: {len(worker_data)} entries")
    print(f"Entrypoint data: {len(entrypoint_times)} entries")
    print(f"Ratio: {len(worker_data) / len(entrypoint_times):.1f}")
    
    # 看看 glen 的成对模式
    print(f"\n=== glen 成对模式 ===")
    print(f"{'pair':>4}  {'glen_a':>8}  {'glen_b':>8}  {'same?':>6}  {'time_a':>8}  {'time_b':>8}  {'total':>10}")
    print("-" * 65)
    
    pair_data = []
    for i in range(0, min(len(worker_data), len(entrypoint_times) * 2), 2):
        pair_idx = i // 2
        if pair_idx >= len(entrypoint_times):
            break
        glen_a, time_a = worker_data[i]
        glen_b, time_b = worker_data[i + 1] if i + 1 < len(worker_data) else (0, 0)
        total = entrypoint_times[pair_idx]
        same = "YES" if glen_a == glen_b else "NO"
        print(f"{pair_idx:>4}  {glen_a:>8.0f}  {glen_b:>8.0f}  {same:>6}  {time_a:>8.0f}  {time_b:>8.0f}  {total:>10.1f}")
        pair_data.append((glen_a, glen_b, time_a, time_b, total))
    
    # 分析：total 是否 ≈ max(time_a, time_b) + overhead?
    print(f"\n=== total vs max(time_a, time_b) ===")
    print(f"{'pair':>4}  {'max_worker':>12}  {'total':>10}  {'overhead':>10}  {'match?':>8}")
    print("-" * 55)
    
    overheads = []
    good_match = 0
    for i, (glen_a, glen_b, time_a, time_b, total) in enumerate(pair_data):
        max_worker = max(time_a, time_b)
        overhead = total - max_worker
        match = "YES" if abs(overhead) < 60 else ""
        if abs(overhead) < 60:
            good_match += 1
        overheads.append(overhead)
        if i < 40 or abs(overhead) > 100:  # 只打前 40 个和异常的
            print(f"{i:>4}  {max_worker:>12.0f}  {total:>10.1f}  {overhead:>10.1f}  {match:>8}")
    
    print(f"\n匹配率 (overhead < 60s): {good_match}/{len(pair_data)} = {good_match/len(pair_data)*100:.1f}%")
    print(f"Overhead 统计: mean={statistics.mean(overheads):.1f}s, median={statistics.median(overheads):.1f}s")
    print(f"  stdev={statistics.stdev(overheads):.1f}s, max={max(overheads):.1f}s, min={min(overheads):.1f}s")

    # 关键分析：glen 差异 vs spike
    print(f"\n=== glen 差异 vs spike 分析 ===")
    glen_diffs = []
    time_diffs = []
    for glen_a, glen_b, time_a, time_b, total in pair_data:
        glen_diff = abs(glen_a - glen_b)
        time_diff = abs(time_a - time_b)
        glen_diffs.append(glen_diff)
        time_diffs.append(time_diff)
    
    # 相关性
    mx = statistics.mean(glen_diffs)
    my = statistics.mean(time_diffs)
    sx = sum((x - mx)**2 for x in glen_diffs) ** 0.5
    sy = sum((y - my)**2 for y in time_diffs) ** 0.5
    if sx > 0 and sy > 0:
        r = sum((x - mx) * (y - my) for x, y in zip(glen_diffs, time_diffs)) / (sx * sy)
        print(f"Pearson r(|glen_a - glen_b|, |time_a - time_b|) = {r:.4f}")
    
    # total vs max_glen
    max_glens = [max(d[0], d[1]) for d in pair_data]
    totals = [d[4] for d in pair_data]
    mx2 = statistics.mean(max_glens)
    my2 = statistics.mean(totals)
    sx2 = sum((x - mx2)**2 for x in max_glens) ** 0.5
    sy2 = sum((y - my2)**2 for y in totals) ** 0.5
    if sx2 > 0 and sy2 > 0:
        r2 = sum((x - mx2) * (y - my2) for x, y in zip(max_glens, totals)) / (sx2 * sy2)
        print(f"Pearson r(max(glen_a, glen_b), total_time) = {r2:.4f}")

if __name__ == "__main__":
    main()
