#!/usr/bin/env python3
"""
将预过滤结果 (rollout_results.jsonl) 转换为训练用的 parquet 格式。

预过滤结果只包含 instance_id 和 reward，需要从原始数据集中获取完整样本信息。

用法:
    python scripts/convert_prefilter_to_parquet.py \
        --input output/prefilter_9k_curriculum/rollout_results.jsonl \
        --original data/swe_gym/swe-smith/train.parquet \
        --output data/swesmith_filtered_4b/ \
        --min-reward 0.0 \
        --val-ratio 0.05
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def load_rollout_results(input_path: str) -> list[dict]:
    """加载预过滤结果"""
    results = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def load_original_data(original_path: str) -> pd.DataFrame:
    """加载原始数据集"""
    if original_path.endswith(".parquet"):
        return pd.read_parquet(original_path)
    elif original_path.endswith(".jsonl"):
        data = []
        with open(original_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported format: {original_path}")


def filter_by_reward(results: list[dict], min_reward: float) -> list[dict]:
    """按 reward 过滤"""
    filtered = []
    for r in results:
        reward = r.get("reward", r.get("total_reward", 0))
        if reward is None:
            reward = 0
        if reward >= min_reward:
            filtered.append(r)
    return filtered


def merge_with_original(results: list[dict], original_df: pd.DataFrame) -> pd.DataFrame:
    """将预过滤结果与原始数据合并"""
    # 创建 instance_id -> reward 映射
    reward_map = {}
    turns_map = {}
    for r in results:
        iid = r.get("instance_id", "")
        reward_map[iid] = r.get("reward", r.get("total_reward", 0))
        turns_map[iid] = r.get("num_turns", 0)
    
    # 过滤原始数据
    filtered_ids = set(reward_map.keys())
    filtered_df = original_df[original_df["instance_id"].isin(filtered_ids)].copy()
    
    # 添加预过滤元数据
    filtered_df["_prefilter_reward"] = filtered_df["instance_id"].map(reward_map)
    filtered_df["_prefilter_turns"] = filtered_df["instance_id"].map(turns_map)
    
    return filtered_df


def main():
    parser = argparse.ArgumentParser(description="Convert prefilter results to parquet")
    parser.add_argument("--input", "-i", required=True, help="Input rollout_results.jsonl")
    parser.add_argument("--original", required=True, help="Original dataset (parquet or jsonl)")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--min-reward", type=float, default=0.0, help="Minimum reward threshold")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()
    
    # 加载预过滤结果
    print(f"Loading prefilter results: {args.input}...")
    results = load_rollout_results(args.input)
    print(f"  Loaded {len(results)} samples")
    
    # 过滤
    if args.min_reward > 0:
        print(f"Filtering by reward >= {args.min_reward}...")
        results = filter_by_reward(results, args.min_reward)
        print(f"  Remaining: {len(results)} samples")
    
    # 加载原始数据
    print(f"Loading original dataset: {args.original}...")
    original_df = load_original_data(args.original)
    print(f"  Loaded {len(original_df)} samples")
    
    # 合并
    print("Merging with original data...")
    df = merge_with_original(results, original_df)
    print(f"  Merged: {len(df)} samples")
    
    # 分割训练/验证集
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # shuffle
    
    val_size = int(len(df) * args.val_ratio)
    val_df = df[:val_size]
    train_df = df[val_size:]
    
    print(f"Split: train={len(train_df)}, val={len(val_df)}")
    
    # 保存
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "validation.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    print(f"Saved:")
    print(f"  {train_path} ({len(train_df)} samples)")
    print(f"  {val_path} ({len(val_df)} samples)")
    
    # 统计信息
    if "_prefilter_reward" in df.columns:
        print(f"\nReward distribution:")
        print(f"  Mean: {df['_prefilter_reward'].mean():.3f}")
        print(f"  Median: {df['_prefilter_reward'].median():.3f}")
        print(f"  Min: {df['_prefilter_reward'].min():.3f}")
        print(f"  Max: {df['_prefilter_reward'].max():.3f}")


if __name__ == "__main__":
    main()
