"""Tests for efficiency metrics (token-level only).

Turn/step counting and tool call counting are tested in test_trajectory_metrics.py.
"""

import pytest
from src.metrics.efficiency_metrics import (
    compute_token_metrics,
    compute_all_efficiency_metrics,
)


# Mock data structures
MOCK_TOKEN_MESSAGE_1 = {
    "kind": "TokenEvent",
    "prompt_token_ids": [1, 2, 3, 4],  # 4 tokens
    "response_token_ids": [5, 6, 7],  # 3 tokens
}

MOCK_TOKEN_MESSAGE_2 = {
    "kind": "TokenEvent",
    "prompt_token_ids": [1, 2, 3, 4, 5, 6],  # 6 tokens
    "response_token_ids": [7, 8],  # 2 tokens
}

MOCK_ASSISTANT_MESSAGE_NO_TOOLS = {
    "role": "assistant",
    "content": "Here's the result",
}

MOCK_TOOL_RESPONSE = {
    "role": "tool",
    "content": "Search results here",
}


class TestComputeTokenMetrics:
    """Tests for compute_token_metrics function."""

    def test_empty_messages(self):
        result = compute_token_metrics([])
        assert result["total_tokens"] == 0
        assert result["total_prompt_tokens"] == 0
        assert result["total_response_tokens"] == 0
        assert result["avg_prompt_tokens_per_turn"] == 0.0
        assert result["avg_response_tokens_per_turn"] == 0.0

    def test_no_token_events(self):
        messages = [MOCK_ASSISTANT_MESSAGE_NO_TOOLS, MOCK_TOOL_RESPONSE]
        result = compute_token_metrics(messages)
        assert result["total_tokens"] == 0

    def test_single_token_event(self):
        messages = [MOCK_TOKEN_MESSAGE_1]
        result = compute_token_metrics(messages)
        assert result["total_tokens"] == 7  # 4 prompt + 3 response
        assert result["total_prompt_tokens"] == 4
        assert result["total_response_tokens"] == 3
        assert result["avg_prompt_tokens_per_turn"] == 4.0
        assert result["avg_response_tokens_per_turn"] == 3.0

    def test_multiple_token_events(self):
        messages = [MOCK_TOKEN_MESSAGE_1, MOCK_TOKEN_MESSAGE_2]
        result = compute_token_metrics(messages)
        assert result["total_tokens"] == 15  # (4+3) + (6+2)
        assert result["total_prompt_tokens"] == 10  # 4 + 6
        assert result["total_response_tokens"] == 5  # 3 + 2
        assert result["avg_prompt_tokens_per_turn"] == 5.0  # 10/2
        assert result["avg_response_tokens_per_turn"] == 2.5  # 5/2

    def test_mixed_messages(self):
        messages = [
            MOCK_TOKEN_MESSAGE_1,
            MOCK_ASSISTANT_MESSAGE_NO_TOOLS,
            MOCK_TOKEN_MESSAGE_2,
            MOCK_TOOL_RESPONSE,
        ]
        result = compute_token_metrics(messages)
        assert result["total_tokens"] == 15
        assert result["total_prompt_tokens"] == 10
        assert result["total_response_tokens"] == 5


class TestComputeAllEfficiencyMetrics:
    """Tests for compute_all_efficiency_metrics function."""

    def test_empty_messages(self):
        result = compute_all_efficiency_metrics(
            messages=[],
            wall_clock_duration=10.5,
        )
        assert result["tokens"] == 0
        assert result["wall_clock_duration"] == 10.5
        # steps and tool metrics should NOT be present (handled by trajectory_metrics)
        assert "steps" not in result
        assert "avg_tool_calls_per_step" not in result

    def test_complete_trajectory(self):
        messages = [
            MOCK_TOKEN_MESSAGE_1,  # 7 tokens
            MOCK_TOKEN_MESSAGE_2,  # 8 tokens
        ]
        result = compute_all_efficiency_metrics(
            messages=messages,
            wall_clock_duration=15.5,
        )
        assert result["tokens"] == 15
        assert result["wall_clock_duration"] == 15.5
        assert result["total_prompt_tokens"] == 10
        assert result["total_response_tokens"] == 5
        assert result["avg_prompt_tokens_per_turn"] == 5.0
        assert result["avg_response_tokens_per_turn"] == 2.5

    def test_ignores_extra_kwargs(self):
        """Extra kwargs (e.g. start_timestamp) should not raise."""
        result = compute_all_efficiency_metrics(
            messages=[MOCK_TOKEN_MESSAGE_1],
            wall_clock_duration=5.0,
            start_timestamp="2025-01-01T10:00:00",
            end_timestamp="2025-01-01T10:00:05",
        )
        assert result["tokens"] == 7

    def test_minimal_trajectory(self):
        messages = [MOCK_TOKEN_MESSAGE_1]
        result = compute_all_efficiency_metrics(
            messages=messages,
            wall_clock_duration=2.0,
        )
        assert result["tokens"] == 7
        assert result["wall_clock_duration"] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
