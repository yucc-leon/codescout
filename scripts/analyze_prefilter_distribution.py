#!/usr/bin/env python3
"""
分析 prefilter rollout 数据的 reward 分布，帮助确定合理的课程学习阈值。

用法:
    # 分析已有的 prefilter parquet
    python scripts/analyze_prefilter_distribution.py \
        --input output/curriculum_static_2stage/prefilter/train.parquet

    # 同时分析 rollout_results.jsonl（包含被过滤掉的样本）
    python scripts/analyze_prefilter_distribution.py \
        --input output/curriculum_static_2stage/prefilter/train.parquet \
        --rollout output/curriculum_static_2stage/prefilter/rollout_results.jsonl

    # 模拟不同阈值下各阶段的样本数
    python scripts/analyze_prefilter_distribution.py \
        --input output/curriculum_static_2stage/prefilter/train.parquet \
        --simulate-stages 2
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def analyze_parquet(path: str, reward_col: str = "prefilter_reward"):
    df = pd.read_parquet(path)
    rewards = df[reward_col]
    print(f"\n{'='*70}")
    print(f"Parquet: {path}")
    print(f"{'='*70}")
    print(f"  Total samples  : {len(df)}")
    print(f"  Mean / Std     : {rewards.mean():.4f} / {rewards.std():.4f}")
    print(f"  Min / Max      : {rewards.min():.4f} / {rewards.max():.4f}")
    print(f"  Median         : {rewards.median():.4f}")

    # 分位数
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    print(f"\n  Quantiles:")
    for q in quantiles:
        print(f"    P{int(q*100):02d} = {rewards.quantile(q):.4f}")

    # 按区间统计
    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.01]
    labels = ["0~0.5", "0.5~1.0", "1.0~1.5", "1.5~2.0", "2.0~2.5", "2.5~3.0"]
    print(f"\n  Reward distribution by bin:")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            mask = (rewards >= lo) & (rewards <= hi)
        else:
            mask = (rewards >= lo) & (rewards < hi)
        n = mask.sum()
        pct = 100 * n / len(df)
        bar = "█" * int(pct / 2)
        print(f"    [{lo:.1f}, {hi:.1f}): {n:5d}  ({pct:5.1f}%)  {bar}")

    # 按现有阈值模拟
    print(f"\n  Current stage thresholds (progressive):")
    thresholds = {1: 2.0, 2: 1.0, 3: 0.5, 4: 0.0}
    for stage, lo in thresholds.items():
        n = (rewards > lo).sum()
        print(f"    Stage {stage} (reward > {lo:.1f}): {n:5d}  ({100*n/len(df):.1f}%)")

    return df, rewards


def analyze_rollout_jsonl(path: str):
    results = []
    with open(path) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                pass

    total = len(results)
    errors = [r for r in results if r.get("error") is not None]
    no_finish = [r for r in results if not r.get("called_finish_tool")]
    zero_reward = [r for r in results if r.get("reward", 0) == 0 and r.get("error") is None]
    positive = [r for r in results if r.get("reward", 0) > 0 and r.get("error") is None]

    print(f"\n{'='*70}")
    print(f"Rollout JSONL: {path}")
    print(f"{'='*70}")
    print(f"  Total rollouts       : {total}")
    print(f"  Errors (filtered out): {len(errors)} ({100*len(errors)/total:.1f}%)")
    print(f"  No finish tool       : {len(no_finish)} ({100*len(no_finish)/total:.1f}%)")
    print(f"  Reward == 0 (no err) : {len(zero_reward)} ({100*len(zero_reward)/total:.1f}%)")
    print(f"  Reward > 0 (no err)  : {len(positive)} ({100*len(positive)/total:.1f}%)")

    if errors:
        error_types = {}
        for r in errors:
            err = r.get("error", "")
            key = err.split(":")[0] if ":" in err else err[:50]
            error_types[key] = error_types.get(key, 0) + 1
        print(f"\n  Error breakdown:")
        for k, v in sorted(error_types.items(), key=lambda x: -x[1])[:10]:
            print(f"    {k}: {v}")

    if positive:
        rewards = [r["reward"] for r in positive]
        print(f"\n  Positive reward stats:")
        print(f"    Mean: {np.mean(rewards):.4f}, Std: {np.std(rewards):.4f}")
        print(f"    Min: {np.min(rewards):.4f}, Max: {np.max(rewards):.4f}")

    return results


def simulate_stages(rewards: pd.Series, num_stages: int, batch_size: int = 8, steps_per_stage: int = 100):
    max_samples = steps_per_stage * batch_size
    print(f"\n{'='*70}")
    print(f"Stage simulation: {num_stages} stages, {steps_per_stage} steps/stage, batch={batch_size}")
    print(f"Max samples per stage: {max_samples}")
    print(f"{'='*70}")

    # 方案 1: 当前硬编码阈值
    print(f"\n  [方案 1] 当前硬编码阈值 (progressive):")
    current = {1: 2.0, 2: 1.0, 3: 0.5, 4: 0.0}
    for s in range(1, num_stages + 1):
        if s in current:
            lo = current[s]
            n = (rewards > lo).sum()
            actual = min(n, max_samples)
            steps = actual // batch_size
            print(f"    Stage {s}: reward > {lo:.1f} → {n} available, {actual} used, {steps} steps")

    # 方案 2: 分位数阈值
    print(f"\n  [方案 2] 分位数阈值 (progressive):")
    if num_stages == 2:
        quantile_thresholds = {1: 0.5, 2: 0.0}  # top 50%, then all
    elif num_stages == 3:
        quantile_thresholds = {1: 0.67, 2: 0.33, 3: 0.0}
    else:
        quantile_thresholds = {s: 1.0 - s / num_stages for s in range(1, num_stages + 1)}
    for s, q in quantile_thresholds.items():
        lo = rewards.quantile(q) if q > 0 else rewards.min() - 0.01
        n = (rewards > lo).sum()
        actual = min(n, max_samples)
        steps = actual // batch_size
        print(f"    Stage {s}: reward > {lo:.4f} (P{int(q*100)}) → {n} available, {actual} used, {steps} steps")

    # 方案 3: 均分阈值
    print(f"\n  [方案 3] 均分 reward 范围 (progressive):")
    rmin, rmax = rewards.min(), rewards.max()
    step_size = (rmax - rmin) / num_stages
    for s in range(1, num_stages + 1):
        lo = rmax - s * step_size
        n = (rewards > lo).sum()
        actual = min(n, max_samples)
        steps = actual // batch_size
        print(f"    Stage {s}: reward > {lo:.4f} → {n} available, {actual} used, {steps} steps")


def main():
    parser = argparse.ArgumentParser(description="分析 prefilter reward 分布")
    parser.add_argument("--input", required=True, help="prefilter train.parquet 路径")
    parser.add_argument("--rollout", help="rollout_results.jsonl 路径（可选）")
    parser.add_argument("--reward-col", default="prefilter_reward")
    parser.add_argument("--simulate-stages", type=int, help="模拟 N 阶段课程")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps-per-stage", type=int, default=100)
    args = parser.parse_args()

    df, rewards = analyze_parquet(args.input, args.reward_col)

    if args.rollout and os.path.exists(args.rollout):
        analyze_rollout_jsonl(args.rollout)

    if args.simulate_stages:
        simulate_stages(rewards, args.simulate_stages, args.batch_size, args.steps_per_stage)

    print()


if __name__ == "__main__":
    main()
