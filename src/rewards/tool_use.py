from src.rewards import reward

@reward("tool_use_reward")
def tool_use_reward(messages, **kwargs) -> float:
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]
    
    num_turns = len(token_messages)
    num_tool_calls = len(tool_messages)
    
    if num_turns == 0:
        return 0.0
    
    if num_tool_calls/num_turns >= 1.0:
        return 1.0

    return 0.0

@reward("turn_efficiency")
def turn_efficiency(messages, max_turns=10, **kwargs) -> float:
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]
    
    num_turns = len(token_messages)
    num_tool_calls = len(tool_messages)
    
    if num_turns == 0:
        return 0.0
    
    if (num_tool_calls > 1) and (num_turns < max_turns):
        return 1.0

    return 0.0