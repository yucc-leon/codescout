"""
效率 Reward - 在答案正确的前提下鼓励并行工具调用和简洁回答

设计原则:
1. 条件性奖励: 只有调用了 localization_finish tool 时才给效率奖励，避免 reward hacking
2. 鼓励并行: 同一 turn 调用多个工具有 bonus
3. 鼓励简洁: 少 turn 完成任务有奖励
"""

from collections import defaultdict
from typing import Dict, Any, Tuple, List

from src.rewards import reward


@reward("efficiency_reward")
def efficiency_reward(
    messages: List[Dict[str, Any]],
    structured_locations: list[dict] | None = None,
    max_turns: int = 5,
    turn_efficiency_bonus: float = 0.1,
    parallel_bonus: float = 0.1,
    **kwargs
) -> Tuple[float, Dict[str, Any]]:
    """
    条件性效率 Reward - 只有正确调用 tool 时才给效率奖励
    
    Args:
        messages: 对话消息列表
        structured_locations: 从 localization_finish tool 提取的结构化位置（用于判断是否正确调用了 tool）
        max_turns: 最大 turn 数阈值
        turn_efficiency_bonus: 少 turn 完成的奖励
        parallel_bonus: 并行工具调用的奖励
    
    Returns:
        reward: 效率奖励分数
        details: 详细指标
    """
    details = {
        "efficiency_reward": 0.0,
        "turn_efficiency_bonus": 0.0,
        "parallel_bonus": 0.0,
        "num_turns": 0,
        "num_tool_calls": 0,
        "has_parallel_calls": False,
    }
    
    # 没有调用 localization_finish tool，不给效率奖励（避免 reward hacking）
    if structured_locations is None:
        details["skipped_reason"] = "no_tool_call"
        return 0.0, details
    
    token_messages = [msg for msg in messages if msg.get("kind") == "TokenEvent"]
    action_messages = [msg for msg in messages if msg.get("kind") == "ActionEvent"]
    
    num_turns = len(token_messages)
    num_tool_calls = len(action_messages)
    
    details["num_turns"] = num_turns
    details["num_tool_calls"] = num_tool_calls
    
    total_reward = 0.0
    
    # 1. 少 turn 奖励
    if num_turns > 0 and num_turns <= max_turns:
        total_reward += turn_efficiency_bonus
        details["turn_efficiency_bonus"] = turn_efficiency_bonus
    
    # 2. 并行工具调用奖励（同一 turn 多个 tool call）
    calls_by_response = defaultdict(int)
    for action in action_messages:
        llm_response_id = action.get("llm_response_id")
        if llm_response_id:
            calls_by_response[llm_response_id] += 1
    
    # 如果有任何一个 turn 调用了多个工具，给 bonus
    has_parallel = any(count > 1 for count in calls_by_response.values())
    if has_parallel:
        total_reward += parallel_bonus
        details["parallel_bonus"] = parallel_bonus
        details["has_parallel_calls"] = True
    
    details["efficiency_reward"] = total_reward
    return total_reward, details
