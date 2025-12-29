import re
from src.rewards import reward

@reward("format_reward")
def format_reward(
    final_message: str,
    **kwargs
    ):
    
    matches = re.findall(r"```(.*?)```", final_message, re.DOTALL)
    parsed_final_message = matches[0] if matches else final_message

    if parsed_final_message.strip() == "":
        return -10.0, {"format_reward": 0.0}
    else:
        return 1.0, {"format_reward": 1.0}