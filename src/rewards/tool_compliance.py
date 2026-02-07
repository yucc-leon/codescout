"""
Tool Compliance Reward - 惩罚明显的工具使用违规

设计原则:
1. 惩罚导向: 只惩罚明显错误，不奖励"看起来正确"的行为
2. 简单规则: 规则越少，被 hack 的可能性越低
3. 可配置: 支持不同场景的定制

违规类型:
1. 使用未定义的工具
2. 超过并行调用限制
3. 使用过于宽泛的匹配模式
"""

import re
from typing import List, Dict, Any, Tuple, Set
from src.rewards import reward


# 常见的过于宽泛的模式
OVERLY_BROAD_PATTERNS = [
    r'\*\.py',           # *.py
    r'\*\*/\*',          # **/*
    r'\.\s*$',           # 单独的 .
    r'^\*$',             # 单独的 *
    r'--include=\*',     # grep --include=*
]

# 编译正则表达式
BROAD_PATTERN_REGEXES = [re.compile(p) for p in OVERLY_BROAD_PATTERNS]


def extract_tool_calls_from_messages(messages: List[Dict]) -> List[Dict[str, Any]]:
    """从 messages 中提取工具调用信息"""
    tool_calls = []
    
    for msg in messages:
        if msg.get("kind") == "ActionEvent":
            tool_calls.append({
                "tool": msg.get("tool", "unknown"),
                "args": msg.get("args", {}),
                "turn": msg.get("turn", 0),
            })
    
    return tool_calls


def check_broad_patterns(args: Dict) -> bool:
    """检查工具参数是否包含过于宽泛的模式"""
    # 将所有参数值转为字符串检查
    args_str = str(args)
    
    for regex in BROAD_PATTERN_REGEXES:
        if regex.search(args_str):
            return True
    
    return False


def compute_tool_compliance_penalty(
    messages: List[Dict],
    allowed_tools: Set[str] = None,
    max_calls_per_turn: int = 5,
    unknown_tool_penalty: float = 0.1,
    excess_calls_penalty: float = 0.05,
    broad_pattern_penalty: float = 0.1,
) -> Tuple[float, Dict[str, Any]]:
    """
    计算工具使用违规惩罚
    
    Args:
        messages: 消息列表
        allowed_tools: 允许的工具集合，None 表示不检查
        max_calls_per_turn: 每 turn 最大调用次数
        unknown_tool_penalty: 使用未知工具的惩罚
        excess_calls_penalty: 超过调用限制的惩罚（每超一次）
        broad_pattern_penalty: 使用宽泛模式的惩罚
    
    Returns:
        penalty: 总惩罚值（负数）
        details: 详细信息
    """
    tool_calls = extract_tool_calls_from_messages(messages)
    
    total_penalty = 0.0
    details = {
        "total_tool_calls": len(tool_calls),
        "unknown_tool_count": 0,
        "excess_calls_count": 0,
        "broad_pattern_count": 0,
    }
    
    # 按 turn 分组
    calls_by_turn: Dict[int, List[Dict]] = {}
    for call in tool_calls:
        turn = call["turn"]
        if turn not in calls_by_turn:
            calls_by_turn[turn] = []
        calls_by_turn[turn].append(call)
    
    for turn, calls in calls_by_turn.items():
        # 1. 检查是否超过并行调用限制
        if len(calls) > max_calls_per_turn:
            excess = len(calls) - max_calls_per_turn
            total_penalty += excess_calls_penalty * excess
            details["excess_calls_count"] += excess
        
        for call in calls:
            # 2. 检查是否使用了未定义的工具
            if allowed_tools is not None and call["tool"] not in allowed_tools:
                total_penalty += unknown_tool_penalty
                details["unknown_tool_count"] += 1
            
            # 3. 检查是否使用了过于宽泛的模式
            if check_broad_patterns(call["args"]):
                total_penalty += broad_pattern_penalty
                details["broad_pattern_count"] += 1
    
    details["total_penalty"] = total_penalty
    
    return -total_penalty, details  # 返回负值


@reward("tool_compliance_penalty")
def tool_compliance_penalty_fn(
    messages: List[Dict],
    allowed_tools: str = "bash,glob,grep",  # 逗号分隔的工具列表
    max_calls_per_turn: int = 5,
    unknown_tool_penalty: float = 0.1,
    excess_calls_penalty: float = 0.05,
    broad_pattern_penalty: float = 0.1,
    **kwargs
) -> Tuple[float, Dict[str, Any]]:
    """
    工具合规性惩罚 Reward
    
    注意: 这是一个惩罚函数，返回值 <= 0
    应该与主要的 localization reward 组合使用
    """
    # 解析允许的工具列表
    allowed_set = set(t.strip() for t in allowed_tools.split(",")) if allowed_tools else None
    
    penalty, details = compute_tool_compliance_penalty(
        messages=messages,
        allowed_tools=allowed_set,
        max_calls_per_turn=max_calls_per_turn,
        unknown_tool_penalty=unknown_tool_penalty,
        excess_calls_penalty=excess_calls_penalty,
        broad_pattern_penalty=broad_pattern_penalty,
    )
    
    return penalty, details


@reward("tool_compliance_soft")
def tool_compliance_soft_fn(
    messages: List[Dict],
    **kwargs
) -> Tuple[float, Dict[str, Any]]:
    """
    软性工具合规惩罚 - 只检查宽泛模式，不检查工具类型
    
    适用于: 不确定具体允许哪些工具的场景
    """
    return tool_compliance_penalty_fn(
        messages=messages,
        allowed_tools=None,  # 不检查工具类型
        max_calls_per_turn=5,
        unknown_tool_penalty=0.0,
        excess_calls_penalty=0.05,
        broad_pattern_penalty=0.15,  # 稍微加重宽泛模式的惩罚
        **kwargs
    )
