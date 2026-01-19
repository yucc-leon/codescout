from src.rewards import reward

@reward("tool_use_reward")
def tool_use_reward(messages, max_tool_use=5, penalize=False, clamp=False, reduction="mean", **kwargs) -> float:
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]
    
    num_turns = max(1, len(token_messages) - 1)
    num_tool_calls = len(tool_messages)

    if num_tool_calls == 0:
        if penalize:
            return -5.0
        else:
            return 0.0
    
    if reduction == "mean":
        average_tool_use = num_tool_calls / num_turns
        reward = min(average_tool_use, max_tool_use) / max_tool_use
    else:
        reward = min(num_tool_calls, max_tool_use) / max_tool_use

    if clamp:
        if reward > 0:
            return 1.0
        else:
            return 0.0
    else:
        return reward

@reward("turn_efficiency")
def turn_efficiency(messages, max_turns=5, **kwargs) -> float:
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]
    
    num_turns = len(token_messages)
    num_tool_calls = len(tool_messages)
    
    if num_turns <= 1:
        return 0.0
    
    if (num_tool_calls > 1):
        # Decay the reward if more than max_turns are used
        if num_turns <= max_turns:
            return 1.0
        else:
            return max(0.0, 1.0 - (num_turns - max_turns) * 0.1)

    return 0.0