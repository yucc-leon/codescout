#!/usr/bin/env python3
"""
课程学习阶段样本选择脚本

根据预过滤的 reward 分数选择不同难度的样本用于各训练阶段。

用法:
    # 选择阶段 1 (简单) 样本
    python scripts/select_curriculum_stage.py \
        --input output/prefilter_9k_curriculum/train.parquet \
        --stage 1 \
        --output output/curriculum/stage1_data.parquet
    
    # 自定义难度阈值
    python scripts/select_curriculum_stage.py \
        --input output/prefilter_9k_curriculum/train.parquet \
        --min-reward 1.5 --max-reward 2.5 \
        --output output/curriculum/custom_data.parquet
    
    # 查看难度分布
    python scripts/select_curriculum_stage.py \
        --input output/prefilter_9k_curriculum/train.parquet \
        --analyze-only
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


# 默认阶段划分（基于 1000 样本预过滤的分布）
DEFAULT_STAGE_THRESHOLDS = {
    # stage: (min_reward, max_reward)  - 左开右闭 (min, max]
    1: (2.0, float('inf')),   # 简单: reward > 2.0, ~33%
    2: (1.0, 2.0),            # 中等: 1.0 < reward <= 2.0, ~22%
    3: (0.5, 1.0),            # 较难: 0.5 < reward <= 1.0, ~25%
    4: (0.0, 0.5),            # 困难: 0 < reward <= 0.5, ~20%
}

# 渐进式阶段划分（每阶段包含之前所有难度）
PROGRESSIVE_STAGE_THRESHOLDS = {
    1: (2.0, float('inf')),   # 只有简单
    2: (1.0, float('inf')),   # 简单 + 中等
    3: (0.5, float('inf')),   # 简单 + 中等 + 较难
    4: (0.0, float('inf')),   # 全部
}


def parse_args():
    parser = argparse.ArgumentParser(description="课程学习阶段样本选择")
    parser.add_argument("--input", type=str, required=True,
                        help="输入数据路径 (parquet 格式，需包含 prefilter_reward 列)")
    parser.add_argument("--output", type=str,
                        help="输出数据路径")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4],
                        help="训练阶段 (1=简单, 2=中等, 3=较难, 4=困难)")
    parser.add_argument("--progressive", action="store_true",
                        help="使用渐进式阶段划分（每阶段包含之前所有难度）")
    parser.add_argument("--min-reward", type=float,
                        help="自定义最小 reward 阈值")
    parser.add_argument("--max-reward", type=float,
                        help="自定义最大 reward 阈值")
    parser.add_argument("--reward-col", type=str, default="prefilter_reward",
                        help="reward 列名")
    parser.add_argument("--max-samples", type=int,
                        help="最大样本数（从选中样本中随机采样）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--analyze-only", action="store_true",
                        help="只分析难度分布，不输出数据")
    return parser.parse_args()


def analyze_distribution(df: pd.DataFrame, reward_col: str = "prefilter_reward"):
    """分析 reward 分布"""
    rewards = df[reward_col]
    
    print("\n" + "=" * 60)
    print("Reward Distribution Analysis")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Mean reward: {rewards.mean():.3f}")
    print(f"Std reward: {rewards.std():.3f}")
    print(f"Min reward: {rewards.min():.3f}")
    print(f"Max reward: {rewards.max():.3f}")
    
    print("\nDistribution by stage:")
    for stage, (low, high) in DEFAULT_STAGE_THRESHOLDS.items():
        mask = (rewards > low) & (rewards <= high)
        count = mask.sum()
        pct = 100 * count / len(df)
        high_str = f"{high:.1f}" if high != float('inf') else "∞"
        print(f"  Stage {stage} ({low:.1f} < r <= {high_str}): {count} ({pct:.1f}%)")
    
    print("\nHistogram:")
    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    hist, _ = pd.cut(rewards, bins=bins, retbins=True)
    for interval, count in hist.value_counts().sort_index().items():
        pct = 100 * count / len(df)
        bar = "█" * int(pct / 2)
        print(f"  {interval}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print("=" * 60)


def select_samples(
    df: pd.DataFrame,
    min_reward: float,
    max_reward: float,
    reward_col: str = "prefilter_reward",
    max_samples: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    """选择指定 reward 范围的样本"""
    mask = (df[reward_col] > min_reward) & (df[reward_col] <= max_reward)
    selected = df[mask].copy()
    
    if max_samples and len(selected) > max_samples:
        selected = selected.sample(n=max_samples, random_state=seed)
    
    return selected


def main():
    args = parse_args()
    
    # 加载数据
    print(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} samples")
    
    # 检查 reward 列
    if args.reward_col not in df.columns:
        raise ValueError(f"Column '{args.reward_col}' not found. Available: {list(df.columns)}")
    
    # 分析模式
    if args.analyze_only:
        analyze_distribution(df, args.reward_col)
        return
    
    # 确定阈值
    if args.min_reward is not None and args.max_reward is not None:
        min_reward = args.min_reward
        max_reward = args.max_reward
    elif args.stage:
        thresholds = PROGRESSIVE_STAGE_THRESHOLDS if args.progressive else DEFAULT_STAGE_THRESHOLDS
        min_reward, max_reward = thresholds[args.stage]
    else:
        raise ValueError("Must specify either --stage or both --min-reward and --max-reward")
    
    # 选择样本
    selected = select_samples(
        df,
        min_reward=min_reward,
        max_reward=max_reward,
        reward_col=args.reward_col,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    
    max_str = f"{max_reward:.1f}" if max_reward != float('inf') else "∞"
    print(f"\nSelected {len(selected)} samples with {min_reward:.1f} < reward <= {max_str}")
    
    if len(selected) == 0:
        print("Warning: No samples selected!")
        return
    
    # 统计
    rewards = selected[args.reward_col]
    print(f"  Mean reward: {rewards.mean():.3f}")
    print(f"  Min reward: {rewards.min():.3f}")
    print(f"  Max reward: {rewards.max():.3f}")
    
    # 保存
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        selected.to_parquet(args.output)
        print(f"\nSaved to {args.output}")
        
        # 保存元信息
        meta = {
            "source": args.input,
            "stage": args.stage,
            "min_reward": min_reward,
            "max_reward": max_reward if max_reward != float('inf') else "inf",
            "progressive": args.progressive,
            "total_samples": len(selected),
            "mean_reward": float(rewards.mean()),
        }
        meta_path = args.output.replace(".parquet", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
