from src.rewards import reward

from src.rewards.file_localization.file_localization import (
    multilevel_localization_f1_reward,
    file_localization_f1_reward
    )

@reward("scaled_f1_reward")
def scaled_f1_reward(
    final_message,
    messages,
    instance,
    multilevel=False,
    **kwargs
    ):

    try:
        if multilevel:
            loc_reward, reward_dict = multilevel_localization_f1_reward(final_message, instance, **kwargs)
        else:
            loc_reward, reward_dict = file_localization_f1_reward(final_message, instance, **kwargs)

    except Exception as e:
        print(f"Error computing localization reward: {e}")
        loc_reward = 0.0
        reward_dict = {
            "multilevel_localization_f1_reward": 0.0,
            "file_reward": 0.0,
            "module_reward": 0.0,
            "entity_reward": 0.0,
        }

    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]
    
    num_turns = len(token_messages) - 1
    if num_turns == 0:
        num_turns = 1  # to avoid division by zero

    num_tool_calls = len(tool_messages)

    avg_tool_calls_per_turn = num_tool_calls / num_turns if num_turns > 0 else 0
    if avg_tool_calls_per_turn > 5:
        avg_tool_calls_per_turn = 5  # cap at ideal avg tool calls

    avg_tool_calls_per_turn = avg_tool_calls_per_turn / 5  # normalize by ideal avg tool calls

    reward_dict["tool_use_reward"] = avg_tool_calls_per_turn

    # Penalize if no tool calls were made
    if avg_tool_calls_per_turn <= 0:
        reward = -5
        return reward, reward_dict

    reward = loc_reward * avg_tool_calls_per_turn
    
    return reward, reward_dict