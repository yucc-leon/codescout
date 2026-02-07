"""
Efficiency metrics — token usage only.

Turn/step counting and tool call counting are handled by trajectory_metrics.py
to avoid duplication. This module focuses on token-level statistics.
"""

from typing import Dict, List, Any


def compute_token_metrics(messages: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute token usage metrics from messages.

    Args:
        messages: List of message dictionaries from conversation events

    Returns:
        Dictionary with keys:
        - total_tokens: Total tokens (prompt + response)
        - total_prompt_tokens: Total prompt tokens
        - total_response_tokens: Total response/completion tokens
        - avg_prompt_tokens_per_turn: Average prompt tokens per TokenEvent
        - avg_response_tokens_per_turn: Average response tokens per TokenEvent
    """
    token_messages = [msg for msg in messages if msg.get("kind") == "TokenEvent"]

    if not token_messages:
        return {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "avg_prompt_tokens_per_turn": 0.0,
            "avg_response_tokens_per_turn": 0.0,
        }

    total_prompt_tokens = sum(len(msg.get("prompt_token_ids", [])) for msg in token_messages)
    total_response_tokens = sum(len(msg.get("response_token_ids", [])) for msg in token_messages)

    num_turns = len(token_messages)
    avg_prompt = total_prompt_tokens / num_turns if num_turns > 0 else 0.0
    avg_response = total_response_tokens / num_turns if num_turns > 0 else 0.0

    return {
        "total_tokens": total_prompt_tokens + total_response_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_response_tokens": total_response_tokens,
        "avg_prompt_tokens_per_turn": avg_prompt,
        "avg_response_tokens_per_turn": avg_response,
    }


def compute_all_efficiency_metrics(
    messages: List[Dict[str, Any]],
    wall_clock_duration: float,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute token-level efficiency metrics for a trajectory.

    Turn counts, tool call counts, and per-turn tool stats are provided by
    trajectory_metrics.compute_trajectory_metrics() — not duplicated here.

    Args:
        messages: List of message dictionaries from conversation events
        wall_clock_duration: Duration in seconds

    Returns:
        Dictionary containing token efficiency metrics
    """
    token_metrics = compute_token_metrics(messages)

    return {
        "tokens": token_metrics["total_tokens"],
        "wall_clock_duration": wall_clock_duration,
        "total_prompt_tokens": token_metrics["total_prompt_tokens"],
        "total_response_tokens": token_metrics["total_response_tokens"],
        "avg_prompt_tokens_per_turn": token_metrics["avg_prompt_tokens_per_turn"],
        "avg_response_tokens_per_turn": token_metrics["avg_response_tokens_per_turn"],
    }
