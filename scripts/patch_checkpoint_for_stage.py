#!/usr/bin/env python3
"""
Stage Transition Checkpoint Patcher
====================================
解决课程学习（curriculum training）阶段切换时 skyrl 丢失训练状态的问题。

问题根因
--------
当前 Stop-and-Swap 方案在切换阶段时用 HF exported weights 重启训练，导致：
  - Adam 一阶/二阶矩（m₁, m₂）全部丢失
  - LR scheduler 位置重置
  - 前 N 步相当于从头训，与课程学习的初衷背道而驰

skyrl 的 fully_async_trainer.load_checkpoints() 有一个关键短路：
  if global_step == 0:
      return 0, checkpoint_path, None   # 跳过 consumed_uids assert！

本脚本利用这一特性：
  1. 复制 Stage N 末尾 checkpoint 的 policy/ 目录（含 optimizer state + LR scheduler）
     到 Stage N+1 的 global_step_0/ 目录
  2. 写入 global_step=0 的 trainer_state.pt（触发短路，绕开 consumed_uids assert）
  3. 不复制 data.pt / fully_async_state.pt（dataloader 从头开始，consumed UIDs 为空）
  4. 创建 latest_ckpt_global_step.txt

Stage N+1 使用 resume_mode="from_path" 加载 global_step_0 检查点：
  - 模型权重：来自 Stage N 末尾 ✓
  - Adam moments（m₁, m₂）：来自 Stage N 末尾 ✓
  - LR scheduler 位置：延续 Stage N 的值 ✓
  - dataloader / consumed_uids：清空，Stage N+1 数据从头读 ✓

用法
----
  uv run python scripts/patch_checkpoint_for_stage.py \\
      --src-ckpt-dir  <stage_N_ckpt_dir>    \\   # e.g. ckpts/curriculum_s1/
      --dst-ckpt-dir  <stage_N+1_ckpt_dir>  \\   # e.g. ckpts/curriculum_s2/
      [--src-step N]                              # 默认：自动找最新 step

  # 查看 src 目录中可用的 checkpoint 步骤
  uv run python scripts/patch_checkpoint_for_stage.py \\
      --src-ckpt-dir ckpts/curriculum_s1/ --list-steps
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

GLOBAL_STEP_PREFIX = "global_step_"
LATEST_CKPT_FILE = "latest_ckpt_global_step.txt"


def find_available_steps(ckpt_dir: Path) -> list[int]:
    steps = []
    for d in ckpt_dir.iterdir():
        if d.is_dir() and d.name.startswith(GLOBAL_STEP_PREFIX):
            try:
                step = int(d.name[len(GLOBAL_STEP_PREFIX):])
                policy_dir = d / "policy"
                # Only include steps that have at least one model shard
                if policy_dir.exists() and any(
                    f.name.startswith("model_world_size_") for f in policy_dir.iterdir()
                ):
                    steps.append(step)
            except ValueError:
                pass
    return sorted(steps)


def find_latest_step(ckpt_dir: Path) -> int:
    latest_file = ckpt_dir / LATEST_CKPT_FILE
    if latest_file.exists():
        try:
            step = int(latest_file.read_text().strip())
            policy_dir = ckpt_dir / f"{GLOBAL_STEP_PREFIX}{step}" / "policy"
            if policy_dir.exists():
                return step
            print(f"[warn] latest_ckpt_global_step.txt points to step {step} "
                  f"but policy dir not found, scanning...")
        except ValueError:
            pass

    steps = find_available_steps(ckpt_dir)
    if not steps:
        raise FileNotFoundError(f"No valid checkpoint found in {ckpt_dir}")
    return max(steps)


def patch_checkpoint(src_ckpt_dir: str, dst_ckpt_dir: str, src_step: int | None = None):
    src = Path(src_ckpt_dir).resolve()
    dst = Path(dst_ckpt_dir).resolve()

    if not src.exists():
        print(f"ERROR: src checkpoint dir not found: {src}")
        sys.exit(1)

    # Find source step
    if src_step is None:
        src_step = find_latest_step(src)
        print(f"[info] Auto-selected source step: {src_step}")
    else:
        policy_dir = src / f"{GLOBAL_STEP_PREFIX}{src_step}" / "policy"
        if not policy_dir.exists():
            print(f"ERROR: policy dir not found at step {src_step}: {policy_dir}")
            sys.exit(1)

    src_step_dir = src / f"{GLOBAL_STEP_PREFIX}{src_step}"
    src_policy_dir = src_step_dir / "policy"
    src_trainer_state = src_step_dir / "trainer_state.pt"

    print(f"[info] Source checkpoint: {src_step_dir}")
    print(f"[info] Destination:       {dst}")

    # Prepare destination global_step_0 directory
    dst.mkdir(parents=True, exist_ok=True)
    dst_step0_dir = dst / f"{GLOBAL_STEP_PREFIX}0"
    dst_policy_dir = dst_step0_dir / "policy"

    if dst_step0_dir.exists():
        print(f"[warn] Destination global_step_0 already exists: {dst_step0_dir}")
        print("[warn] Removing and recreating...")
        shutil.rmtree(dst_step0_dir)

    dst_step0_dir.mkdir(parents=True)

    # ── Step 1: Copy policy/ (model weights + optimizer state + LR scheduler) ──
    print(f"[copy] policy/ ({src_policy_dir} → {dst_policy_dir})")
    shutil.copytree(src_policy_dir, dst_policy_dir)
    n_files = sum(1 for _ in dst_policy_dir.rglob("*") if _.is_file())
    print(f"       Copied {n_files} files")

    # ── Step 2: Write trainer_state.pt with global_step=0 ──────────────────────
    # Load original config (if available) and reuse it; only reset global_step.
    # This ensures model architecture / tokenizer config matches.
    if src_trainer_state.exists():
        orig_state = torch.load(src_trainer_state, map_location="cpu", weights_only=False)
        orig_config = orig_state.get("config", None)
        print(f"[info] Loaded original trainer config from step {src_step}")
    else:
        orig_config = None
        print("[warn] trainer_state.pt not found at src step; creating minimal state")

    new_trainer_state = {
        "global_step": 0,
        "config": orig_config,
        # Stage annotation for debugging
        "_patched_from_step": src_step,
        "_patched_from_dir": str(src_step_dir),
    }
    dst_trainer_state = dst_step0_dir / "trainer_state.pt"
    torch.save(new_trainer_state, dst_trainer_state)
    print(f"[write] trainer_state.pt → global_step=0 (patched from step {src_step})")

    # ── Step 3: Intentionally skip data.pt and fully_async_state.pt ───────────
    # - data.pt missing → StatefulDataLoader logs warning and starts from beginning ✓
    # - fully_async_state.pt missing → load_checkpoints returns global_step=0 early,
    #   bypassing the consumed_uids assert entirely ✓
    print("[skip] data.pt              (dataloader will restart from beginning)")
    print("[skip] fully_async_state.pt (consumed UIDs cleared, assert bypassed)")

    # ── Step 4: Write latest_ckpt_global_step.txt ───────────────────────────────
    latest_file = dst / LATEST_CKPT_FILE
    latest_file.write_text("0")
    print(f"[write] latest_ckpt_global_step.txt → 0")

    print()
    print("=" * 60)
    print(f"Checkpoint patch complete!")
    print(f"  Source step    : {src_step}  ({src_step_dir})")
    print(f"  Destination    : {dst_step0_dir}")
    print()
    print("Usage in training:")
    print(f'  trainer.resume_mode="from_path"')
    print(f'  trainer.resume_path="{dst_step0_dir}"')
    print()
    print("What is preserved:")
    print("  ✓ Model weights (from Stage N end)")
    print("  ✓ Adam moments m₁, m₂ (gradient statistics, most critical)")
    print("  ✓ LR scheduler position (continues from Stage N's final LR)")
    print()
    print("What is reset (intentional):")
    print("  ✗ global_step → 0 (so LR scheduler/epoch logic works for new stage)")
    print("  ✗ consumed_uids → [] (Stage N+1 data treated as fresh)")
    print("  ✗ dataloader state → resets to start of Stage N+1 dataset)")
    print("=" * 60)

    return str(dst_step0_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Patch skyrl checkpoint for curriculum stage transition "
                    "(preserves optimizer state, clears dataloader state)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src-ckpt-dir", required=True,
        help="Source checkpoint directory (Stage N training output, e.g. ckpts/curriculum_s1/)"
    )
    parser.add_argument(
        "--dst-ckpt-dir", required=True,
        help="Destination checkpoint directory (Stage N+1 will resume from here)"
    )
    parser.add_argument(
        "--src-step", type=int, default=None,
        help="Which global_step to use from src-ckpt-dir (default: auto-detect latest)"
    )
    parser.add_argument(
        "--list-steps", action="store_true",
        help="List available steps in src-ckpt-dir and exit"
    )
    args = parser.parse_args()

    src = Path(args.src_ckpt_dir).resolve()

    if args.list_steps:
        steps = find_available_steps(src)
        if not steps:
            print(f"No valid checkpoints found in {src}")
        else:
            latest = find_latest_step(src)
            print(f"Available steps in {src}:")
            for s in steps:
                marker = " ← latest" if s == latest else ""
                print(f"  global_step_{s}{marker}")
        return

    patch_checkpoint(
        src_ckpt_dir=args.src_ckpt_dir,
        dst_ckpt_dir=args.dst_ckpt_dir,
        src_step=args.src_step,
    )


if __name__ == "__main__":
    main()
