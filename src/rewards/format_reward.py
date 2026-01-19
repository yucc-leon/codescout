import re
from src.rewards import reward

@reward("format_reward")
def format_reward(
    final_message: str,
    START_STRING: str = "```",
    END_STRING: str = "```",
    penalize: bool = True,
    **kwargs
    ):
    
    final_message = final_message.strip()
    if final_message.startswith(START_STRING) and END_STRING in final_message:
        return 1.0, {"format_reward": 1.0}
    else:
        if penalize:
            return -5.0, {"format_reward": -5.0}
        else:
            return 0.0, {"format_reward": 0.0}
