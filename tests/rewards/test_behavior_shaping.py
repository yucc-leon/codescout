"""Tests for behavior shaping reward."""

import pytest
from src.rewards.behavior_shaping import _compute_behavior_shaping


def _make_messages(num_turns, tool_calls_per_turn=None):
    """Helper to build mock messages.

    Args:
        num_turns: number of TokenEvent messages
        tool_calls_per_turn: list of ints, tool calls per turn.
            If None, no ActionEvents are created.
    """
    msgs = []
    for i in range(num_turns):
        msgs.append({"kind": "TokenEvent", "prompt_token_ids": [1], "response_token_ids": [2]})
        if tool_calls_per_turn and i < len(tool_calls_per_turn):
            for _ in range(tool_calls_per_turn[i]):
                msgs.append({"kind": "ActionEvent", "tool_name": "bash", "llm_response_id": f"resp_{i}"})
    return msgs


class TestPenalties:
    """Penalties should always fire regardless of correctness."""

    def test_turn_overrun(self):
        msgs = _make_messages(num_turns=6, tool_calls_per_turn=[1]*6)
        reward, details = _compute_behavior_shaping(
            msgs, structured_locations=[{"file": "a.py"}],
            correctness_reward=0.0, max_turns=4, max_parallel_calls=5,
        )
        assert details["bs_turn_overrun"] == 1.0
        assert reward < 0

    def test_parallel_overrun(self):
        msgs = _make_messages(num_turns=2, tool_calls_per_turn=[7, 1])
        reward, details = _compute_behavior_shaping(
            msgs, structured_locations=[{"file": "a.py"}],
            correctness_reward=0.5, max_turns=4, max_parallel_calls=5,
        )
        assert details["bs_parallel_overrun"] == 1.0
        assert details["bs_penalty"] == -0.05

    def test_no_finish(self):
        msgs = _make_messages(num_turns=4, tool_calls_per_turn=[1]*4)
        reward, details = _compute_behavior_shaping(
            msgs, structured_locations=None,
            correctness_reward=0.0, max_turns=4, max_parallel_calls=5,
        )
        assert details["bs_no_finish"] == 1.0

    def test_no_penalty_when_compliant(self):
        msgs = _make_messages(num_turns=3, tool_calls_per_turn=[2, 3, 1])
        reward, details = _compute_behavior_shaping(
            msgs, structured_locations=[{"file": "a.py"}],
            correctness_reward=0.0, max_turns=4, max_parallel_calls=5,
        )
        assert details["bs_penalty"] == 0.0


class TestBonus:
    """Bonus should only fire when correctness > 0."""

    def test_bonus_with_correctness(self):
        msgs = _make_messages(num_turns=3, tool_calls_per_turn=[2, 1, 1])
        reward, details = _compute_behavior_shaping(
            msgs, structured_locations=[{"file": "a.py"}],
            correctness_reward=0.8, max_turns=4, max_parallel_calls=5,
        )
        assert details["bs_efficient_finish"] == 1.0
        assert details["bs_bonus"] == 0.05
        assert reward == 0.05  # no penalty + bonus

    def test_no_bonus_without_correctness(self):
        msgs = _make_messages(num_turns=3, tool_calls_per_turn=[2, 1, 1])
        reward, details = _compute_behavior_shaping(
            msgs, structured_locations=[{"file": "a.py"}],
            correctness_reward=0.0, max_turns=4, max_parallel_calls=5,
        )
        assert details["bs_efficient_finish"] == 0.0
        assert details["bs_bonus"] == 0.0
        assert reward == 0.0

    def test_no_bonus_when_over_budget(self):
        msgs = _make_messages(num_turns=5, tool_calls_per_turn=[1]*5)
        reward, details = _compute_behavior_shaping(
            msgs, structured_locations=[{"file": "a.py"}],
            correctness_reward=1.0, max_turns=4, max_parallel_calls=5,
        )
        # Over budget: penalty fires, no bonus
        assert details["bs_turn_overrun"] == 1.0
        assert details["bs_efficient_finish"] == 0.0


class TestCombined:
    """Test combined penalty + bonus scenarios."""

    def test_perfect_trajectory(self):
        """Correct, within budget, no violations."""
        msgs = _make_messages(num_turns=3, tool_calls_per_turn=[3, 2, 1])
        reward, details = _compute_behavior_shaping(
            msgs, structured_locations=[{"file": "a.py"}],
            correctness_reward=1.0, max_turns=4, max_parallel_calls=5,
        )
        assert details["bs_penalty"] == 0.0
        assert details["bs_bonus"] == 0.05
        assert reward == 0.05

    def test_worst_case(self):
        """All violations, no correctness."""
        msgs = _make_messages(num_turns=6, tool_calls_per_turn=[8, 1, 1, 1, 1, 1])
        reward, details = _compute_behavior_shaping(
            msgs, structured_locations=None,
            correctness_reward=0.0, max_turns=4, max_parallel_calls=5,
        )
        # turn_overrun + parallel_overrun + no_finish
        assert reward == -0.10 + -0.05 + -0.10
        assert details["bs_bonus"] == 0.0

    def test_empty_messages(self):
        reward, details = _compute_behavior_shaping(
            messages=[], structured_locations=None,
            correctness_reward=0.0, max_turns=4, max_parallel_calls=5,
        )
        assert details["bs_num_turns"] == 0
        assert reward == 0.0

    def test_reward_range(self):
        """Total reward should be in [-0.25, +0.05]."""
        # Best case
        msgs = _make_messages(num_turns=2, tool_calls_per_turn=[1, 1])
        best, _ = _compute_behavior_shaping(
            msgs, structured_locations=[{"file": "a.py"}],
            correctness_reward=1.0, max_turns=4, max_parallel_calls=5,
        )
        assert best == 0.05

        # Worst case
        msgs = _make_messages(num_turns=6, tool_calls_per_turn=[8, 1, 1, 1, 1, 1])
        worst, _ = _compute_behavior_shaping(
            msgs, structured_locations=None,
            correctness_reward=0.0, max_turns=4, max_parallel_calls=5,
        )
        assert worst == -0.25
