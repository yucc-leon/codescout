"""
Length-normalized advantage estimators for GRPO.

This module implements three variants of length normalization:
1. sqrt(length): Proposed method, balances bias reduction and stability
2. linear(length): Stronger normalization, divides by actual length
3. log(length): Weaker normalization, uses logarithm of length

These estimators address the sequence length bias in step-wise=false training,
where longer sequences receive disproportionately large gradients.
"""

from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
from skyrl_train.utils.ppo_utils import register_advantage_estimator


@register_advantage_estimator("grpo_length_norm_sqrt")
def compute_grpo_length_norm_sqrt_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO with sqrt(length) normalization.
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"No score in prompt index: {idx}")
        seq_lengths = response_mask.sum(dim=-1)
        for i in range(bsz):
            adv = scores[i] - id2mean[index[i]]
            length_factor = torch.sqrt(seq_lengths[i].float() + epsilon)
            scores[i] = adv / length_factor
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores


@register_advantage_estimator("grpo_length_norm_linear")
def compute_grpo_length_norm_linear_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO with linear length normalization.
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"No score in prompt index: {idx}")
        seq_lengths = response_mask.sum(dim=-1)
        for i in range(bsz):
            adv = scores[i] - id2mean[index[i]]
            scores[i] = adv / (seq_lengths[i].float() + epsilon)
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores


@register_advantage_estimator("grpo_length_norm_log")
def compute_grpo_length_norm_log_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO with log(length) normalization.
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"No score in prompt index: {idx}")
        seq_lengths = response_mask.sum(dim=-1)
        for i in range(bsz):
            adv = scores[i] - id2mean[index[i]]
            length_factor = torch.log(seq_lengths[i].float() + 1.0 + epsilon)
            scores[i] = adv / length_factor
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores
