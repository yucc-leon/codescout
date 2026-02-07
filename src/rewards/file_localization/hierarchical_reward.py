"""
通用层级化 Localization Reward

设计原则:
1. 层级依赖: 上层错误影响下层得分
2. 召回优先: 漏报比误报代价更高
3. 粒度奖励: 更细粒度的正确定位获得更高奖励
4. 可配置: 支持不同任务场景的定制

参考:
- Agentless: 三阶段 localization → repair → validation
- SweRank: retrieve-and-rerank, Top-K 评估
- 数据分析: ~53% 单 module, ~45% 单 entity, ~4% GT 为空
"""

from typing import List, Set, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .module_rewards import get_simple_results_from_raw_outputs, parse_structured_outputs
from src.rewards import reward


class GatingStrategy(Enum):
    """层级门控策略"""
    NONE = "none"                    # 无门控，各层独立
    SOFT = "soft"                    # 软门控: score_i * prev_score
    HARD = "hard"                    # 硬门控: score_i if prev > θ else 0
    CASCADING = "cascading"          # 级联: score_i * Π(prev_scores)


@dataclass
class LevelConfig:
    """单层配置"""
    name: str
    weight: float = 1.0
    beta: float = 1.0               # F_beta 的 beta 值，>1 偏向召回
    exact_match_bonus: float = 0.0  # 完全匹配额外奖励


@dataclass 
class HierarchicalRewardConfig:
    """层级 Reward 配置"""
    levels: List[LevelConfig] = field(default_factory=lambda: [
        LevelConfig("file", weight=1.0, beta=1.5),
        LevelConfig("module", weight=1.0, beta=2.0),
        LevelConfig("entity", weight=1.0, beta=2.0),
    ])
    gating: GatingStrategy = GatingStrategy.SOFT
    gate_threshold: float = 0.5     # for HARD gating
    abstain_bonus: float = 0.1      # GT 为空时正确不预测的奖励
    over_pred_penalty: float = 0.0  # 过度预测惩罚系数


def compute_f_beta(pred: Set, gt: Set, beta: float = 1.0) -> float:
    """
    计算 F_beta 分数
    
    beta > 1: 偏向召回 (推荐 beta=2 for localization)
    beta < 1: 偏向精确
    beta = 1: 标准 F1
    """
    if not gt:
        return 1.0 if not pred else 0.0
    
    tp = len(pred & gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt)
    
    if precision + recall == 0:
        return 0.0
    
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def compute_recall(pred: Set, gt: Set) -> float:
    """纯召回率"""
    if not gt:
        return 1.0 if not pred else 0.0
    return len(pred & gt) / len(gt)


def compute_over_prediction_ratio(pred: Set, gt: Set) -> float:
    """过度预测比例: 预测数量超过 GT 的部分"""
    if not gt:
        return len(pred)  # GT 为空时，任何预测都是过度
    return max(0, len(pred) - len(gt)) / len(gt)


def hierarchical_localization_reward(
    predictions: List[Set],
    ground_truths: List[Set],
    config: HierarchicalRewardConfig,
) -> Tuple[float, Dict[str, Any]]:
    """
    计算层级化 localization reward
    
    Args:
        predictions: [pred_files, pred_modules, pred_entities]
        ground_truths: [gt_files, gt_modules, gt_entities]
        config: 配置
    
    Returns:
        total_reward: 总分
        details: 各层详细得分 (核心指标 + detailed_metrics/ 前缀的详细指标)
    """
    assert len(predictions) == len(ground_truths) == len(config.levels)
    
    n_levels = len(config.levels)
    level_scores = []
    details = {}
    
    # 1. 计算各层原始得分
    for i, (pred, gt, level_cfg) in enumerate(zip(predictions, ground_truths, config.levels)):
        score = compute_f_beta(pred, gt, level_cfg.beta)
        level_scores.append(score)
        
        # 核心指标 - 统一命名，与 multilevel_localization_f1_reward 对齐
        details[f"{level_cfg.name}_reward"] = score
        
        # 完全匹配 bonus
        exact_match = 1.0 if (pred == gt and level_cfg.exact_match_bonus > 0) else 0.0
        details[f"detailed_metrics/{level_cfg.name}_exact_match"] = exact_match
    
    # 2. 计算门控值
    level_gates = [1.0] * n_levels  # 第一层无门控
    
    if config.gating == GatingStrategy.SOFT:
        for i in range(1, n_levels):
            level_gates[i] = level_scores[i-1]
    
    elif config.gating == GatingStrategy.HARD:
        for i in range(1, n_levels):
            level_gates[i] = 1.0 if level_scores[i-1] >= config.gate_threshold else 0.0
    
    elif config.gating == GatingStrategy.CASCADING:
        for i in range(1, n_levels):
            level_gates[i] = np.prod(level_scores[:i])
    
    # 3. 计算加权得分
    total_reward = 0.0
    for i, (score, gate, level_cfg) in enumerate(zip(level_scores, level_gates, config.levels)):
        gated_score = score * gate
        
        # 加入 exact match bonus
        exact_match = details[f"detailed_metrics/{level_cfg.name}_exact_match"]
        exact_bonus = level_cfg.exact_match_bonus * exact_match
        
        weighted_score = level_cfg.weight * (gated_score + exact_bonus)
        total_reward += weighted_score
        
        # 详细指标 - 放到 detailed_metrics/ 下
        details[f"detailed_metrics/{level_cfg.name}_gate"] = gate
        details[f"detailed_metrics/{level_cfg.name}_gated_score"] = gated_score
        details[f"detailed_metrics/{level_cfg.name}_weighted_score"] = weighted_score
    
    # 4. Abstain bonus (GT 为空时正确不预测)
    for i, (pred, gt, level_cfg) in enumerate(zip(predictions, ground_truths, config.levels)):
        if len(gt) == 0 and len(pred) == 0:
            total_reward += config.abstain_bonus
            details[f"detailed_metrics/{level_cfg.name}_abstain_bonus"] = config.abstain_bonus
    
    # 5. 过度预测惩罚
    if config.over_pred_penalty > 0:
        total_penalty = 0.0
        for i, (pred, gt, level_cfg) in enumerate(zip(predictions, ground_truths, config.levels)):
            over_ratio = compute_over_prediction_ratio(pred, gt)
            penalty = config.over_pred_penalty * over_ratio * level_cfg.weight
            total_penalty += penalty
            details[f"detailed_metrics/{level_cfg.name}_over_pred_penalty"] = penalty
        
        total_reward -= total_penalty
        details["detailed_metrics/total_over_pred_penalty"] = total_penalty
    
    # 核心指标
    details["total_reward"] = total_reward
    return total_reward, details


# ============ 预定义配置 ============

def get_swe_bench_config() -> HierarchicalRewardConfig:
    """SWE-bench 推荐配置 - 为下游 patch 生成优化"""
    return HierarchicalRewardConfig(
        levels=[
            LevelConfig("file", weight=1.2, beta=1.5),
            LevelConfig("module", weight=1.0, beta=2.0),
            LevelConfig("entity", weight=0.8, beta=2.0),
        ],
        gating=GatingStrategy.SOFT,
        abstain_bonus=0.1,
        over_pred_penalty=0.0,
    )


def get_recall_focused_config() -> HierarchicalRewardConfig:
    """召回优先配置 - 绝对不能漏"""
    return HierarchicalRewardConfig(
        levels=[
            LevelConfig("file", weight=1.0, beta=3.0),  # 强召回
            LevelConfig("module", weight=1.0, beta=2.0),
            LevelConfig("entity", weight=1.0, beta=2.0),
        ],
        gating=GatingStrategy.CASCADING,
        abstain_bonus=0.1,
        over_pred_penalty=0.0,
    )


def get_balanced_config() -> HierarchicalRewardConfig:
    """平衡配置 - 通用场景"""
    return HierarchicalRewardConfig(
        levels=[
            LevelConfig("file", weight=1.0, beta=1.5, exact_match_bonus=0.2),
            LevelConfig("module", weight=1.0, beta=1.5, exact_match_bonus=0.2),
            LevelConfig("entity", weight=1.0, beta=1.5, exact_match_bonus=0.3),
        ],
        gating=GatingStrategy.SOFT,
        abstain_bonus=0.1,
        over_pred_penalty=0.1,
    )


def get_general_config() -> HierarchicalRewardConfig:
    """
    一般性配置 - 平衡精确与召回
    
    设计原则:
    1. 层级依赖: 软门控体现 file → module → entity 的包含关系
    2. 平衡: beta=1 标准 F1，不偏向精确或召回
    3. 权重相等: 不做任务特定假设，各层同等重要
    4. 无额外奖惩: 不加 exact_match_bonus 或 over_pred_penalty
    
    数学形式:
        reward = Σ_i (F1(pred_i, gt_i) × gate_i)
        gate_i = score_{i-1}  (软门控)
    """
    return HierarchicalRewardConfig(
        levels=[
            LevelConfig("file", weight=1.0, beta=1.0),
            LevelConfig("module", weight=1.0, beta=1.0),
            LevelConfig("entity", weight=1.0, beta=1.0),
        ],
        gating=GatingStrategy.SOFT,
        abstain_bonus=0.0,      # 不额外奖励
        over_pred_penalty=0.0,  # 不额外惩罚
    )


def get_no_gating_config() -> HierarchicalRewardConfig:
    """
    无门控配置 - 消融实验用
    
    各层独立计算，不考虑层级依赖关系
    用于验证门控机制的作用
    """
    return HierarchicalRewardConfig(
        levels=[
            LevelConfig("file", weight=1.0, beta=1.0),
            LevelConfig("module", weight=1.0, beta=1.0),
            LevelConfig("entity", weight=1.0, beta=1.0),
        ],
        gating=GatingStrategy.NONE,
        abstain_bonus=0.0,
        over_pred_penalty=0.0,
    )


def get_cascading_config() -> HierarchicalRewardConfig:
    """
    级联门控配置 - 消融实验用
    
    score_i * Π(prev_scores)，更严格的层级依赖
    """
    return HierarchicalRewardConfig(
        levels=[
            LevelConfig("file", weight=1.0, beta=1.0),
            LevelConfig("module", weight=1.0, beta=1.0),
            LevelConfig("entity", weight=1.0, beta=1.0),
        ],
        gating=GatingStrategy.CASCADING,
        abstain_bonus=0.0,
        over_pred_penalty=0.0,
    )


# ============ Reward 函数注册 ============

@reward("hierarchical_localization_reward")
def hierarchical_localization_reward_fn(
    final_message: str,
    instance: dict,
    config_name: str = "swe_bench",
    structured_locations: list[dict] | None = None,
    tool_call_bonus: float = 0.1,
    **kwargs
) -> Tuple[float, Dict[str, Any]]:
    """
    层级化 Localization Reward
    
    Args:
        final_message: 模型输出
        instance: 数据实例
        config_name: 配置名称 ("swe_bench", "recall_focused", "balanced", "general")
        structured_locations: 从 localization_finish tool 调用中提取的结构化位置信息
        tool_call_bonus: 正确调用 tool 的 exploration bonus（默认 0.1）
    """
    # 如果没有调用 localization_finish tool，直接返回 0
    if structured_locations is None:
        return 0.0, {
            "file_reward": 0.0,
            "module_reward": 0.0,
            "entity_reward": 0.0,
            "total_reward": 0.0,
            "tool_call_bonus": 0.0,
            "detailed_metrics/config_name": config_name,
            "detailed_metrics/no_tool_call": True,
            "detailed_metrics/pred_file_count": 0,
            "detailed_metrics/pred_module_count": 0,
            "detailed_metrics/pred_entity_count": 0,
        }
    
    # 获取配置
    config_map = {
        "swe_bench": get_swe_bench_config,
        "recall_focused": get_recall_focused_config,
        "balanced": get_balanced_config,
        "general": get_general_config,
        "no_gating": get_no_gating_config,
        "cascading": get_cascading_config,
    }
    config = config_map.get(config_name, get_general_config)()
    
    # 解析 GT
    gt_files = []
    gt_modules = []
    gt_entities = []
    
    for change in instance.get("file_changes", []):
        if isinstance(change, dict):
            if "file" in change:
                gt_files.append(change["file"])
            if "changes" in change:
                edited_modules = change["changes"].get("edited_modules")
                edited_entities = change["changes"].get("edited_entities")
                if edited_modules is not None:
                    gt_modules.extend(list(edited_modules))
                if edited_entities is not None:
                    gt_entities.extend(list(edited_entities))
    
    gt_files = set(gt_files)
    gt_modules = set(gt_modules)
    gt_entities = set(gt_entities)
    
    # 解析预测 - 使用 structured_locations
    pred_files, pred_modules, pred_entities = parse_structured_outputs(structured_locations)
    pred_files = set(pred_files)
    pred_modules = set(pred_modules)
    pred_entities = set(pred_entities)
    
    # 计算 reward
    predictions = [pred_files, pred_modules, pred_entities]
    ground_truths = [gt_files, gt_modules, gt_entities]
    
    total_reward, details = hierarchical_localization_reward(predictions, ground_truths, config)
    
    # 加上 tool call bonus（鼓励模型学会调用 tool）
    total_reward += tool_call_bonus
    details["tool_call_bonus"] = tool_call_bonus
    details["total_reward"] = total_reward
    
    # 详细信息 - 放到 detailed_metrics/ 下
    details["detailed_metrics/config_name"] = config_name
    details["detailed_metrics/no_tool_call"] = False
    details["detailed_metrics/pred_file_count"] = len(pred_files)
    details["detailed_metrics/pred_module_count"] = len(pred_modules)
    details["detailed_metrics/pred_entity_count"] = len(pred_entities)
    details["detailed_metrics/gt_file_count"] = len(gt_files)
    details["detailed_metrics/gt_module_count"] = len(gt_modules)
    details["detailed_metrics/gt_entity_count"] = len(gt_entities)
    
    return total_reward, details


@reward("hierarchical_recall_reward")
def hierarchical_recall_reward_fn(
    final_message: str,
    instance: dict,
    structured_locations: list[dict] | None = None,
    **kwargs
) -> Tuple[float, Dict[str, Any]]:
    """召回优先的层级 Reward"""
    return hierarchical_localization_reward_fn(
        final_message, instance, config_name="recall_focused", 
        structured_locations=structured_locations, **kwargs
    )


@reward("hierarchical_balanced_reward")
def hierarchical_balanced_reward_fn(
    final_message: str,
    instance: dict,
    structured_locations: list[dict] | None = None,
    **kwargs
) -> Tuple[float, Dict[str, Any]]:
    """平衡的层级 Reward"""
    return hierarchical_localization_reward_fn(
        final_message, instance, config_name="balanced",
        structured_locations=structured_locations, **kwargs
    )
