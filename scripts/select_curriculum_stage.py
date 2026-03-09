#!/usr/bin/env python3
"""
课程学习阶段样本选择脚本

根据预过滤的 reward 分数选择不同难度的样本用于各训练阶段。

用法:
    # 查看难度分布（先跑这个，确认阈值合理）
    python scripts/select_curriculum_stage.py \
        --input output/prefilter/train.parquet \
        --analyze-only

    # 选择阶段 1 (简单，渐进式)
    python scripts/select_curriculum_stage.py \
        --input output/prefilter/train.parquet \
        --stage 1 --progressive \
        --output output/curriculum/stage1/train.parquet

    # 自定义阈值
    python scripts/select_curriculum_stage.py \
        --input output/prefilter/train.parquet \
        --min-reward 1.5 \
        --output output/curriculum/custom/train.parquet
"""

import argparse
import json
import os

import pandas as pd

# 默认阶段划分（独立式，每阶段只含该难度区间）
DEFAULT_STAGE_THRESHOLDS = {
    1: (2.0, 3.0),           # easy:      2.0 < reward < 3.0
    2: (1.0, 2.0),           # medium:    1.0 < reward <= 2.0
    3: (0.5, 1.0),           # hard:      0.5 < reward <= 1.0
    4: (0.0, 0.5),           # very hard: 0   < reward <= 0.5
}

# 渐进式阶段划分（每阶段包含该难度及以上所有样本）
PROGRESSIVE_STAGE_THRESHOLDS = {
    1: (2.0, 3.0),           # only easy
    2: (1.0, 3.0),           # easy + medium
    3: (0.5, 3.0),           # easy + medium + hard
    4: (0.0, 3.0),           # all
}


def parse_args():
    parser = argparse.ArgumentParser(description="课程学习阶段样本选择")
    parser.add_argument("--input", required=True, help="输入 parquet 路径（需含 prefilter_reward 列）")
    parser.add_argument("--output", help="输出 parquet 路径")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="训练阶段 (1=easy, 4=very hard)")
    parser.add_argument("--progressive", action="store_true", help="渐进式：每阶段包含之前所有难度")
    parser.add_argument("--min-reward", type=float, help="自定义最小 reward 阈值")
    parser.add_argument("--max-reward", type=float, help="自定义最大 reward 阈值")
    parser.add_argument("--reward-col", default="prefilter_reward", help="reward 列名")
    parser.add_argument("--max-samples", type=int, help="最大样本数（随机采样）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze-only", action="store_true", help="只分析分布，不输出数据")
    return parser.parse_args()


def analyze_distribution(df: pd.DataFrame, reward_col: str):
    rewards = df[reward_col]
    print(f"\n{'='*60}")
    print("Reward Distribution Analysis")
    print(f"{'='*60}")
    print(f"Total samples : {len(df)}")
    print(f"Mean / Std    : {rewards.mean():.3f} / {rewards.std():.3f}")
    print(f"Min / Max     : {rewards.min():.3f} / {rewards.max():.3f}")
    print("\nBy stage (independent):")
    for stage, (lo, hi) in DEFAULT_STAGE_THRESHOLDS.items():
        mask = (rewards > lo) & (rewards < hi)
        n = mask.sum()
        print(f"  Stage {stage} ({lo:.1f} < r < {hi:.1f}): {n:5d}  ({100*n/len(df):.1f}%)")
    print(f"{'='*60}\n")


def select_samples(df, min_reward, max_reward, reward_col, max_samples=None, seed=42):
    mask = (df[reward_col] > min_reward) & (df[reward_col] < max_reward)
    selected = df[mask].copy()
    if max_samples and len(selected) > max_samples:
        selected = selected.sample(n=max_samples, random_state=seed)
    return selected


def main():
    args = parse_args()

    print(f"Loading {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} samples")

    if args.reward_col not in df.columns:
        raise ValueError(f"Column '{args.reward_col}' not found. Available: {list(df.columns)}")

    if args.analyze_only:
        analyze_distribution(df, args.reward_col)
        return

    # 确定阈值
    if args.min_reward is not None:
        min_r = args.min_reward
        max_r = args.max_reward if args.max_reward is not None else float("inf")
    elif args.stage:
        thresholds = PROGRESSIVE_STAGE_THRESHOLDS if args.progressive else DEFAULT_STAGE_THRESHOLDS
        min_r, max_r = thresholds[args.stage]
    else:
        raise ValueError("需要指定 --stage 或 --min-reward")

    selected = select_samples(df, min_r, max_r, args.reward_col, args.max_samples, args.seed)

    max_str = f"{max_r:.1f}" if max_r != float("inf") else "∞"
    print(f"Selected {len(selected)} samples  ({min_r:.1f} < reward < {max_str})")

    if len(selected) == 0:
        print("WARNING: 0 samples selected — check thresholds with --analyze-only")
        return

    rewards = selected[args.reward_col]
    print(f"  mean={rewards.mean():.3f}  min={rewards.min():.3f}  max={rewards.max():.3f}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        selected.to_parquet(args.output, index=False)
        print(f"Saved to {args.output}")

        meta = {
            "source": args.input,
            "stage": args.stage,
            "progressive": args.progressive,
            "min_reward": min_r,
            "max_reward": max_r if max_r != float("inf") else "inf",
            "n_samples": len(selected),
            "mean_reward": float(rewards.mean()),
        }
        meta_path = args.output.replace(".parquet", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
