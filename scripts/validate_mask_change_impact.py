#!/usr/bin/env python3
"""
Validate training impact of mask logic change in code_search_generator.py.

Compared logic:
1) Old (before commit 3bbbf64 in soni/main range):
   found_role_switch = True if token_id == end_token_id else False
2) New (after commit 3bbbf64):
   if token_id == start_token_id:
       found_role_switch = True
   else:
       found_role_switch = False

This script uses mock token ids and reports how many tokens remain trainable
(mask == 1) under each logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple


START_TOKEN_ID = 1001  # mock id for "<|im_start|>"
END_TOKEN_ID = 1002  # mock id for "assistant"


@dataclass
class MaskStats:
    name: str
    num_sequences: int
    avg_trainable_ratio: float
    min_trainable_ratio: float
    max_trainable_ratio: float
    total_trainable_tokens: int
    total_tokens: int


def build_mask(token_ids: List[int], use_new_logic: bool) -> List[int]:
    """Replicate mask generation loop from code_search_generator.py."""
    buffer_succeed = 5
    buffer_precede = 1

    mask: List[int] = []
    inside = False
    buffer = 0
    found_role_switch = False

    for token_id in token_ids:
        if token_id == START_TOKEN_ID:
            inside = True
            for _ in range(buffer_precede):
                if mask:
                    mask.pop()
            mask.extend([0] * buffer_precede)
            mask.append(0)
        elif token_id == END_TOKEN_ID and found_role_switch:
            inside = False
            mask.append(0)
            buffer = buffer_succeed
        else:
            if inside:
                mask.append(0)
            elif buffer:
                mask.append(0)
                buffer -= 1
            else:
                mask.append(1)

        # Logic difference under test.
        if use_new_logic:
            # New logic in soni/main range commit 3bbbf64
            if token_id == START_TOKEN_ID:
                found_role_switch = True
            else:
                found_role_switch = False
        else:
            # Old logic
            found_role_switch = token_id == END_TOKEN_ID

    return mask


def make_mock_sequence(rng: random.Random, turns: int) -> List[int]:
    """
    Build a mock response token stream with repeated role markers and content.
    Pattern per turn (simplified):
      preamble -> <|im_start|> assistant -> 8~20 content tokens
    """
    seq: List[int] = []

    # Some prefix tokens from previous context
    seq.extend([rng.randint(10, 999) for _ in range(rng.randint(3, 10))])

    for _ in range(turns):
        seq.append(START_TOKEN_ID)
        seq.append(END_TOKEN_ID)
        content_len = rng.randint(8, 20)
        seq.extend([rng.randint(10, 999) for _ in range(content_len)])

        # Optional random token noise between turns
        if rng.random() < 0.35:
            seq.extend([rng.randint(10, 999) for _ in range(rng.randint(1, 4))])

    return seq


def summarize(name: str, masks: List[List[int]]) -> MaskStats:
    ratios: List[float] = []
    total_trainable = 0
    total_tokens = 0
    for m in masks:
        trainable = sum(m)
        total = len(m)
        ratios.append(trainable / total if total else 0.0)
        total_trainable += trainable
        total_tokens += total
    return MaskStats(
        name=name,
        num_sequences=len(masks),
        avg_trainable_ratio=sum(ratios) / len(ratios),
        min_trainable_ratio=min(ratios),
        max_trainable_ratio=max(ratios),
        total_trainable_tokens=total_trainable,
        total_tokens=total_tokens,
    )


def main() -> None:
    rng = random.Random(42)

    sequences = [make_mock_sequence(rng, turns=rng.randint(2, 6)) for _ in range(100)]

    old_masks = [build_mask(seq, use_new_logic=False) for seq in sequences]
    new_masks = [build_mask(seq, use_new_logic=True) for seq in sequences]

    old_stats = summarize("old_logic", old_masks)
    new_stats = summarize("new_logic", new_masks)

    print("=== Mock Validation: code_search_generator mask logic impact ===")
    print(f"Samples: {len(sequences)} sequences")
    print("")

    for s in (old_stats, new_stats):
        print(f"[{s.name}]")
        print(f"  avg_trainable_ratio: {s.avg_trainable_ratio:.4f}")
        print(f"  min_trainable_ratio: {s.min_trainable_ratio:.4f}")
        print(f"  max_trainable_ratio: {s.max_trainable_ratio:.4f}")
        print(
            f"  total_trainable_tokens: {s.total_trainable_tokens}/{s.total_tokens} "
            f"({s.total_trainable_tokens / s.total_tokens:.4f})"
        )
        print("")

    delta = new_stats.avg_trainable_ratio - old_stats.avg_trainable_ratio
    print("[delta: new - old]")
    print(f"  avg_trainable_ratio_delta: {delta:.4f}")
    print(
        "  interpretation: positive means new logic keeps more tokens "
        "participating in training loss."
    )
    print("")

    # Show one concrete sequence as an example.
    example_idx = 0
    seq = sequences[example_idx]
    old_mask = old_masks[example_idx]
    new_mask = new_masks[example_idx]
    print("[example sequence #0]")
    print(f"  token_count: {len(seq)}")
    print(f"  old_trainable: {sum(old_mask)}")
    print(f"  new_trainable: {sum(new_mask)}")
    print("  first_40_tokens:", seq[:40])
    print("  old_mask_40:", old_mask[:40])
    print("  new_mask_40:", new_mask[:40])


if __name__ == "__main__":
    main()
