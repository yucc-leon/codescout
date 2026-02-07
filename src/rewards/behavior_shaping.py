"""
Behavior Shaping Reward — 指令遵循奖惩

设计原则:
1. Prompt-Reward 对齐: 每一项奖惩都对应 system prompt 中的明确指令
2. 条件性激活: bonus 仅在精度 reward > 0 时生效（防 reward hacking）
3. 惩罚始终生效: 违反 prompt 指令无论精度如何都应被惩罚
4. 量级远小于精度 reward: 总影响 ±0.25，不翻转精度信号
5. 二值化: 违规/合规，不做连续衰减，避免调参
6. 内部系数硬编码: yaml 只暴露一个 weight

Prompt 指令 → Reward 对应关系:
  "You have up to 4 turns"                    → turn_overrun penalty
  "NEVER exceed 5 parallel tool calls"        → parallel_overrun penalty
  "MUST call localization_finish before runs out" → no_finish penalty
  "Complete your search efficiently"           → efficient_finish bonus (gated)

总范围: [-0.25, +0.05]
  惩罚: turn_overrun(-0.10) + parallel_overrun(-0.05) + no_finish(-0.10)
  奖励: efficient_finish(+0.05)，门控于 correctness > 0
"""

from collections import defaultdict
from typing import Dict, Any, List, Tuple

from src.rewards import reward


# 内部系数，不暴露为参数
_TURN_OVERRUN_PENALTY = -0.10
_PARALLEL_OVERRUN_PENALTY = -0.05
_NO_FINISH_PENALTY = -0.10
_EFFICIENT_FINISH_BONUS = 0.05


def _compute_behavior_shaping(
    messages: List[Dict[str, Any]],
    structured_locations: Any,
    correctness_reward: float,
    max_turns: int,
    max_parallel_calls: int,
) -> Tuple[float, Dict[str, Any]]:
    """
    Core logic for behavior shaping reward.

    Args:
        messages: conversation event list
        structured_locations: output from localization_finish tool (None if not called)
        correctness_reward: the correctness (localization F1) reward for this trajectory
        max_turns: prompt-specified turn limit
        max_parallel_calls: prompt-specified parallel call limit per turn
    """
    token_messages = [m for m in messages if m.get("kind") == "TokenEvent"]
    action_messages = [m for m in messages if m.get("kind") == "ActionEvent"]

    num_turns = len(token_messages)
    num_tool_calls = len(action_messages)

    # --- Penalties (always active) ---

    # 1. Turn overrun: exceeded max_turns
    turn_overrun = num_turns > max_turns
    p_turn = _TURN_OVERRUN_PENALTY if turn_overrun else 0.0

    # 2. Parallel overrun: any single turn exceeded max_parallel_calls
    calls_by_response = defaultdict(int)
    for action in action_messages:
        resp_id = action.get("llm_response_id")
        if resp_id:
            calls_by_response[resp_id] += 1
    parallel_overrun = any(c > max_parallel_calls for c in calls_by_response.values())
    p_parallel = _PARALLEL_OVERRUN_PENALTY if parallel_overrun else 0.0

    # 3. No finish: exhausted all turns without calling finish tool
    no_finish = structured_locations is None and num_turns >= max_turns
    p_no_finish = _NO_FINISH_PENALTY if no_finish else 0.0

    total_penalty = p_turn + p_parallel + p_no_finish

    # --- Bonus (gated on correctness > 0) ---

    has_correctness = correctness_reward > 0
    finished_in_budget = structured_locations is not None and num_turns <= max_turns
    b_efficient = _EFFICIENT_FINISH_BONUS if (has_correctness and finished_in_budget) else 0.0

    total = total_penalty + b_efficient

    details = {
        "behavior_shaping_reward": total,
        "bs_turn_overrun": 1.0 if turn_overrun else 0.0,
        "bs_parallel_overrun": 1.0 if parallel_overrun else 0.0,
        "bs_no_finish": 1.0 if no_finish else 0.0,
        "bs_efficient_finish": 1.0 if b_efficient > 0 else 0.0,
        "bs_penalty": total_penalty,
        "bs_bonus": b_efficient,
        "bs_correctness_gate": 1.0 if has_correctness else 0.0,
        "bs_num_turns": num_turns,
        "bs_num_tool_calls": num_tool_calls,
    }

    return total, details


@reward("behavior_shaping_reward")
def behavior_shaping_reward(
    messages: List[Dict[str, Any]],
    structured_locations: Any = None,
    correctness_reward: float = 0.0,
    max_turns: int = 4,
    max_parallel_calls: int = 5,
    **kwargs,
) -> Tuple[float, Dict[str, Any]]:
    """
    Behavior shaping reward for controlling agent conduct.

    Penalties are always active; bonus is gated on correctness > 0.
    Internal coefficients are fixed — only the outer weight in yaml is tunable.

    Args:
        messages: conversation events
        structured_locations: from localization_finish tool (None = not called)
        correctness_reward: the correctness reward for this trajectory (used for gating)
        max_turns: turn budget from prompt
        max_parallel_calls: parallel tool call limit from prompt
    """
    return _compute_behavior_shaping(
        messages=messages,
        structured_locations=structured_locations,
        correctness_reward=correctness_reward,
        max_turns=max_turns,
        max_parallel_calls=max_parallel_calls,
    )
