"""
Ranking-based reward functions for code localization.

This module implements rewards that consider the ordering of predictions,
not just set-based metrics like F1.

Rewards implemented:
- ndcg_localization_reward: NDCG (Normalized Discounted Cumulative Gain)
- weighted_f1_reward: F-beta score with configurable precision/recall weighting
- precision_at_k_reward: Precision@K for top-k evaluation
"""

import ast
import math
from typing import List, Set, Tuple

from src.rewards import reward
from src.rewards.file_localization.module_rewards import (
    get_simple_results_from_raw_outputs,
    parse_simple_output,
)


def _get_ordered_predictions(final_message: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Parse predictions while preserving order (for ranking metrics).
    
    Returns:
        Tuple of (files, modules, entities) as ordered lists
    """
    locations = parse_simple_output(final_message)
    
    # Preserve order, remove duplicates while keeping first occurrence
    seen_files = set()
    seen_modules = set()
    seen_entities = set()
    
    ordered_files = []
    ordered_modules = []
    ordered_entities = []
    
    for loc in locations:
        file_path = loc["file"]
        class_name = loc.get("class")
        func_name = loc.get("function")
        
        # File
        if file_path not in seen_files:
            seen_files.add(file_path)
            ordered_files.append(file_path)
        
        # Entity and Module
        if func_name:
            if class_name:
                entity = f"{file_path}:{class_name}.{func_name}"
                module = f"{file_path}:{class_name}"
            else:
                entity = f"{file_path}:{func_name}"
                module = entity  # standalone function is its own module
            
            if entity.endswith(".__init__"):
                entity = entity[:-len(".__init__")]
            
            if entity not in seen_entities:
                seen_entities.add(entity)
                ordered_entities.append(entity)
            
            if module not in seen_modules:
                seen_modules.add(module)
                ordered_modules.append(module)
    
    return ordered_files, ordered_modules, ordered_entities


def _get_ground_truth(instance: dict) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extract ground truth files, modules, and entities from instance."""
    gt_files = set()
    gt_modules = set()
    gt_entities = set()
    
    for change in instance.get("file_changes", []):
        if "file" in change:
            gt_files.add(change["file"])
        
        if "changes" in change:
            edited_modules = change["changes"].get("edited_modules") or []
            edited_entities = change["changes"].get("edited_entities") or []
            gt_modules.update(edited_modules)
            gt_entities.update(edited_entities)
    
    return gt_files, gt_modules, gt_entities


def compute_dcg(predictions: List[str], ground_truth: Set[str], k: int = None) -> float:
    """
    Compute Discounted Cumulative Gain.
    
    DCG = sum_{i=1}^{k} rel_i / log2(i + 1)
    where rel_i = 1 if prediction[i] in ground_truth, else 0
    """
    if k is None:
        k = len(predictions)
    
    dcg = 0.0
    for i, pred in enumerate(predictions[:k]):
        if pred in ground_truth:
            # Position is 1-indexed for log calculation
            dcg += 1.0 / math.log2(i + 2)  # +2 because i is 0-indexed
    
    return dcg


def compute_idcg(ground_truth: Set[str], k: int = None) -> float:
    """
    Compute Ideal DCG (best possible DCG).
    
    IDCG assumes all relevant items are ranked at the top.
    """
    n_relevant = len(ground_truth)
    if k is not None:
        n_relevant = min(n_relevant, k)
    
    idcg = 0.0
    for i in range(n_relevant):
        idcg += 1.0 / math.log2(i + 2)
    
    return idcg


@reward("ndcg_localization_reward")
def ndcg_localization_reward(
    final_message: str,
    instance: dict,
    file_level_weight: float = 1.0,
    module_level_weight: float = 0.5,
    entity_level_weight: float = 0.5,
    k: int = None,
    **kwargs
) -> Tuple[float, dict]:
    """
    NDCG-based reward for code localization.
    
    Unlike F1 which only considers set overlap, NDCG rewards correct predictions
    that appear earlier in the output. This encourages the model to:
    1. Put most confident/important files first
    2. Avoid over-reporting (diminishing returns for later items)
    
    Args:
        final_message: Agent's final output
        instance: Problem instance with ground truth
        file_level_weight: Weight for file-level NDCG
        module_level_weight: Weight for module-level NDCG
        entity_level_weight: Weight for entity-level NDCG
        k: Cutoff for NDCG@k (None = use all predictions)
    
    Returns:
        Tuple of (reward_value, metrics_dict)
    """
    # Get ordered predictions
    pred_files, pred_modules, pred_entities = _get_ordered_predictions(final_message)
    
    # Get ground truth
    gt_files, gt_modules, gt_entities = _get_ground_truth(instance)
    
    # Compute NDCG for each level
    def safe_ndcg(predictions, ground_truth, cutoff):
        if not ground_truth:
            # No ground truth - reward empty prediction, penalize non-empty
            return 1.0 if not predictions else 0.0
        if not predictions:
            return 0.0
        
        dcg = compute_dcg(predictions, ground_truth, cutoff)
        idcg = compute_idcg(ground_truth, cutoff)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    file_ndcg = safe_ndcg(pred_files, gt_files, k)
    module_ndcg = safe_ndcg(pred_modules, gt_modules, k)
    entity_ndcg = safe_ndcg(pred_entities, gt_entities, k)
    
    # Weighted combination
    total_reward = (
        file_level_weight * file_ndcg
        + module_level_weight * module_ndcg
        + entity_level_weight * entity_ndcg
    )
    
    return total_reward, {
        "ndcg_reward": total_reward,
        "file_ndcg": file_ndcg,
        "module_ndcg": module_ndcg,
        "entity_ndcg": entity_ndcg,
        "num_pred_files": len(pred_files),
        "num_gt_files": len(gt_files),
    }


@reward("weighted_f1_reward")
def weighted_f1_reward(
    final_message: str,
    instance: dict,
    beta: float = 0.5,
    file_level_weight: float = 1.0,
    module_level_weight: float = 0.5,
    entity_level_weight: float = 0.5,
    **kwargs
) -> Tuple[float, dict]:
    """
    Weighted F-beta score for code localization.
    
    Standard F1 weights precision and recall equally. F-beta allows adjusting
    this balance:
    - beta < 1: Precision is weighted more (penalize false positives)
    - beta = 1: Standard F1
    - beta > 1: Recall is weighted more (penalize false negatives)
    
    For code localization, beta < 1 is often preferred because:
    - Context pollution (false positives) wastes LLM context window
    - Missing files can sometimes be recovered in later steps
    
    Formula: F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
    
    Args:
        final_message: Agent's final output
        instance: Problem instance with ground truth
        beta: F-beta parameter (default 0.5 = precision-weighted)
        file_level_weight: Weight for file-level F-beta
        module_level_weight: Weight for module-level F-beta
        entity_level_weight: Weight for entity-level F-beta
    
    Returns:
        Tuple of (reward_value, metrics_dict)
    """
    # Get predictions (order doesn't matter for F-beta)
    pred_files, pred_modules, pred_entities = get_simple_results_from_raw_outputs(final_message)
    pred_files = set(pred_files)
    pred_modules = set(pred_modules)
    pred_entities = set(pred_entities)
    
    # Get ground truth
    gt_files, gt_modules, gt_entities = _get_ground_truth(instance)
    
    def compute_f_beta(predictions: Set[str], ground_truth: Set[str], beta: float) -> Tuple[float, float, float]:
        """Compute F-beta score and return (f_beta, precision, recall)."""
        if not ground_truth:
            # No ground truth - reward empty prediction
            return (1.0, 1.0, 1.0) if not predictions else (0.0, 0.0, 0.0)
        if not predictions:
            return 0.0, 0.0, 0.0
        
        tp = len(predictions & ground_truth)
        precision = tp / len(predictions)
        recall = tp / len(ground_truth)
        
        if precision + recall == 0:
            return 0.0, 0.0, 0.0
        
        beta_sq = beta ** 2
        f_beta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        
        return f_beta, precision, recall
    
    file_f_beta, file_p, file_r = compute_f_beta(pred_files, gt_files, beta)
    module_f_beta, module_p, module_r = compute_f_beta(pred_modules, gt_modules, beta)
    entity_f_beta, entity_p, entity_r = compute_f_beta(pred_entities, gt_entities, beta)
    
    # Weighted combination
    total_reward = (
        file_level_weight * file_f_beta
        + module_level_weight * module_f_beta
        + entity_level_weight * entity_f_beta
    )
    
    return total_reward, {
        "weighted_f1_reward": total_reward,
        "file_f_beta": file_f_beta,
        "file_precision": file_p,
        "file_recall": file_r,
        "module_f_beta": module_f_beta,
        "entity_f_beta": entity_f_beta,
        "beta": beta,
    }


@reward("precision_at_k_reward")
def precision_at_k_reward(
    final_message: str,
    instance: dict,
    k: int = 5,
    file_level_weight: float = 1.0,
    **kwargs
) -> Tuple[float, dict]:
    """
    Precision@K reward - what fraction of top-k predictions are correct.
    
    This is useful when we only care about the top few predictions,
    which is common in code localization where context window is limited.
    
    Args:
        final_message: Agent's final output
        instance: Problem instance with ground truth
        k: Number of top predictions to consider
        file_level_weight: Weight for file-level P@K
    
    Returns:
        Tuple of (reward_value, metrics_dict)
    """
    # Get ordered predictions
    pred_files, _, _ = _get_ordered_predictions(final_message)
    
    # Get ground truth
    gt_files, _, _ = _get_ground_truth(instance)
    
    if not gt_files:
        return (1.0 if not pred_files else 0.0), {"precision_at_k": 1.0 if not pred_files else 0.0}
    
    # Take top-k predictions
    top_k = pred_files[:k]
    
    if not top_k:
        return 0.0, {"precision_at_k": 0.0, "k": k}
    
    # Compute precision@k
    hits = sum(1 for f in top_k if f in gt_files)
    p_at_k = hits / len(top_k)
    
    return file_level_weight * p_at_k, {
        "precision_at_k": p_at_k,
        "k": k,
        "hits": hits,
        "num_predictions": len(top_k),
    }
