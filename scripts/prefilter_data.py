#!/usr/bin/env python3
"""
数据预过滤脚本 - 课程学习

目的:
1. 验证小模型（4B/14B）能否成功调用 localization_finish tool
2. 过滤出 reward > 0 的样本，用于后续训练
3. 按难度排序（高 reward = 简单），实现课程学习

用法:
    # 完整流程：rollout + 过滤（使用所有 8 个 GPU）
    python scripts/prefilter_data.py --model 4b --output data/swesmith_filtered_4b/
    
    # 小规模测试（验证 prompt 是否有效）
    python scripts/prefilter_data.py --model 4b --max-samples 20 --output data/test_prefilter/
    
    # 从已有 rollout 结果过滤
    python scripts/prefilter_data.py --filter-only --rollout-dir data/swesmith_rollout_4b/ --output data/swesmith_filtered_4b/
    
    # 使用 N-samples（每个 instance 多次 rollout，取最高 reward）
    python scripts/prefilter_data.py --model 4b --n-samples 3 --output data/swesmith_filtered_4b/
    
    # 指定 DP 和 GPU
    python scripts/prefilter_data.py --model 14b --dp-size 4 --visible-gpus 0,1,2,3,4,5,6,7 --output data/swesmith_filtered_14b/

数据集:
    默认使用 adityasoni17/SWE-smith-py-code-search (已预处理的 Python 代码搜索数据集)
    数据路径: data/adityasoni17__SWE-smith-py-code-search_train/train.parquet

GPU 配置:
    - 4B 模型: tp_size=1, 默认 dp_size=8 (8 个 vLLM 实例，每个用 1 个 GPU)
    - 14B 模型: tp_size=2, 默认 dp_size=4 (4 个 vLLM 实例，每个用 2 个 GPU)
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# 禁用 vLLM 的日志和进度条
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig


def parse_args():
    parser = argparse.ArgumentParser(description="数据预过滤脚本 - 课程学习")
    parser.add_argument("--model", type=str, choices=["4b", "14b"], help="模型大小")
    parser.add_argument("--input", type=str, default=None, 
                        help="输入数据路径 (默认: 自动检测 adityasoni17/SWE-smith-py-code-search)")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--config", type=str, default="configs/skyrl-experiments/ablation/prompt_fix_tool_only.yaml", 
                        help="实验配置文件路径")
    parser.add_argument("--rollout-only", action="store_true", help="只做 rollout，不过滤")
    parser.add_argument("--filter-only", action="store_true", help="只做过滤，不 rollout")
    parser.add_argument("--rollout-dir", type=str, help="已有 rollout 结果目录（用于 --filter-only）")
    parser.add_argument("--min-reward", type=float, default=0.0, help="最小 reward 阈值")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--temperature", type=float, default=0.6, help="采样温度")
    parser.add_argument("--max-turns", type=int, default=10, help="最大对话轮数")
    parser.add_argument("--num-workers", type=int, default=8, help="并行 worker 数")
    parser.add_argument("--tp-size", type=int, default=None, help="Tensor parallel size (默认: 4b=1, 14b=2)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU 显存利用率")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计信息，不保存")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续")
    
    # 新增参数：N-samples 和 DP
    parser.add_argument("--n-samples", type=int, default=1, 
                        help="每个 instance 的 rollout 次数 (N)，取最高 reward 的结果")
    parser.add_argument("--dp-size", type=int, default=None,
                        help="数据并行度 (DP)，启动多个 vLLM 实例。默认: 8/tp_size (4b=8, 14b=4)")
    parser.add_argument("--visible-gpus", type=str, default=None,
                        help="可见 GPU 列表，逗号分隔，如 '0,1,2,3,4,5,6,7'")
    parser.add_argument("--shuffle", action="store_true",
                        help="随机打乱数据顺序（推荐，避免数据分布偏差）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（用于 shuffle）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="从训练 checkpoint 加载模型（用于动态课程学习的重新评估）")
    return parser.parse_args()


# ========== 模型配置 ==========

MODEL_CONFIGS = {
    "4b": {
        "model_path": os.environ.get("MODEL_PATH_4B", "models/Qwen3-4B-Instruct-2507"),
        "model_name": "Qwen3-4B-Instruct-2507",
        "tp_size": 1,
    },
    "14b": {
        "model_path": os.environ.get("MODEL_PATH_14B", "models/Qwen3-14B-Instruct-2507"),
        "model_name": "Qwen3-14B-Instruct-2507",
        "tp_size": 2,
    },
}

# 数据集路径候选
DATASET_PATHS = [
    "data/adityasoni17__SWE-smith-py-code-search_train/train.parquet",
    "../acs_pr_bugfix/data/adityasoni17__SWE-smith-py-code-search_train/train.parquet",
]


def find_dataset_path() -> str:
    """自动检测数据集路径"""
    script_dir = Path(__file__).parent.parent
    
    for rel_path in DATASET_PATHS:
        full_path = script_dir / rel_path
        if full_path.exists():
            print(f"Found dataset at: {full_path}")
            return str(full_path)
    
    # 尝试从 HuggingFace 下载
    try:
        from datasets import load_dataset
        print("Dataset not found locally, downloading from HuggingFace...")
        ds = load_dataset("adityasoni17/SWE-smith-py-code-search", split="train")
        output_dir = script_dir / "data" / "adityasoni17__SWE-smith-py-code-search_train"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "train.parquet"
        ds.to_parquet(str(output_path))
        print(f"Downloaded and saved to: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"Failed to download dataset: {e}")
    
    raise FileNotFoundError(
        f"Dataset not found. Please download adityasoni17/SWE-smith-py-code-search "
        f"and place it in one of: {DATASET_PATHS}"
    )


# ========== vLLM 离线推理 ==========

class OfflineVLLMEngine:
    """离线 vLLM 推理引擎
    
    注意：vLLM 的 LLM 类不支持 enable_auto_tool_choice 参数，
    那是 OpenAI 兼容 API server 的参数。
    对于离线推理，我们直接使用 chat template 来处理 tools。
    """
    
    def __init__(
        self,
        model_path: str,
        tp_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 64000,
        device_ids: Optional[List[int]] = None,
        engine_id: int = 0,
    ):
        from vllm import LLM, SamplingParams
        
        self.engine_id = engine_id
        
        # 设置 CUDA_VISIBLE_DEVICES 来指定使用哪些 GPU
        if device_ids is not None:
            cuda_devices = ",".join(str(d) for d in device_ids)
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
            print(f"[Engine {engine_id}] Setting CUDA_VISIBLE_DEVICES={cuda_devices}")
        
        print(f"[Engine {engine_id}] Loading model from {model_path}...")
        print(f"  Tensor parallel size: {tp_size}")
        print(f"  GPU memory utilization: {gpu_memory_utilization}")
        print(f"  Max model len: {max_model_len}")
        
        # RoPE YaRN 扩展配置（与训练保持一致）
        rope_scaling = {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 32768,
        }
        print(f"  RoPE scaling: {rope_scaling}")
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            rope_scaling=rope_scaling,
            disable_log_stats=True,  # 禁用统计日志
        )
        self.tokenizer = self.llm.get_tokenizer()
        print(f"[Engine {engine_id}] Model loaded successfully!")
    
    def generate(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: float = 0.6,
        max_tokens: int = 4096,
    ) -> str:
        """生成单个响应"""
        from vllm import SamplingParams
        
        # 构建 prompt，使用 chat template 处理 tools
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["<|im_end|>"],
        )
        
        # 禁用 vLLM 的进度条
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text


# ========== Rollout 逻辑 ==========

def get_tool_definitions() -> List[Dict]:
    """获取工具定义"""
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Execute a bash command in the terminal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "localization_finish",
                "description": """Submit your final code localization results.

Use this tool when you have identified all relevant files, classes, and functions that need to be modified.

Provide a structured list of locations. Each location must have:
- file: Path to the file relative to the root of the repository (required)
- class_name: Class name (optional)
- function_name: Function/method name (optional)""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "locations": {
                            "type": "array",
                            "description": "List of code locations to modify",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string", "description": "Path to the file"},
                                    "class_name": {"type": "string", "description": "Class name (optional)"},
                                    "function_name": {"type": "string", "description": "Function/method name (optional)"}
                                },
                                "required": ["file"]
                            }
                        }
                    },
                    "required": ["locations"]
                }
            }
        }
    ]


def load_system_prompt(config_path: str) -> str:
    """加载 system prompt"""
    config = OmegaConf.load(config_path)
    prompts_base_dir = Path(__file__).parent.parent / "src" / "prompts"
    system_prompt_path = prompts_base_dir / config.prompts.system_prompt
    
    with open(system_prompt_path) as f:
        return f.read()


def load_user_prompt_template(config_path: str) -> str:
    """加载 user prompt 模板"""
    config = OmegaConf.load(config_path)
    prompts_base_dir = Path(__file__).parent.parent / "src" / "prompts"
    user_prompt_path = prompts_base_dir / config.prompts.user_prompt
    
    with open(user_prompt_path) as f:
        return f.read()


def format_user_prompt(template: str, instance: Dict, working_dir: str) -> str:
    """格式化 user prompt"""
    from jinja2 import Template
    t = Template(template)
    return t.render(instance=instance, working_dir=working_dir)


def parse_tool_call(response: str) -> Optional[Dict]:
    """解析工具调用"""
    import re
    
    # 查找 <tool_call> 标签
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
    if not match:
        return None
    
    try:
        tool_call = json.loads(match.group(1))
        return tool_call
    except json.JSONDecodeError:
        return None


def execute_terminal(command: str, working_dir: str, timeout: int = 30) -> str:
    """执行终端命令"""
    import subprocess
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        return output[:4000]  # 限制输出长度
    except subprocess.TimeoutExpired:
        return "[Command timed out]"
    except Exception as e:
        return f"[Error: {str(e)}]"


def rollout_single_instance(
    instance: Dict,
    engine: OfflineVLLMEngine,
    system_prompt: str,
    user_prompt_template: str,
    tools: List[Dict],
    temperature: float,
    max_turns: int,
    exp_config: DictConfig,
) -> Dict[str, Any]:
    """对单个 instance 进行 rollout"""
    
    # Add src to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils.instance import clone_instance
    from src.rewards import get_reward_function
    
    instance_id = instance["instance_id"]
    repo_name = instance["repo"]
    commit_id = instance.get("base_commit", None)
    patch = instance.get("patch") if instance.get("use_patch") else None
    
    result = {
        "instance_id": instance_id,
        "reward": 0.0,
        "reward_details": {},
        "structured_locations": None,
        "num_turns": 0,
        "called_finish_tool": False,
        "error": None,
        "wall_clock_duration": 0.0,
    }
    
    # Clone repository
    uuid_str = str(uuid.uuid4())[:8]
    workspace = Path(f"/tmp/testbed/{uuid_str}/")
    
    try:
        status, working_dir = clone_instance(repo_name, commit_id, instance_id, workspace, patch)
        if not status:
            result["error"] = f"Failed to clone repository {repo_name}"
            return result
    except Exception as e:
        result["error"] = f"Clone error: {str(e)}"
        return result
    
    start_time = time.time()
    
    try:
        # 构建初始消息
        user_prompt = format_user_prompt(user_prompt_template, instance, str(working_dir))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        structured_locations = None
        
        # 多轮对话
        for turn in range(max_turns):
            result["num_turns"] = turn + 1
            
            # 生成响应
            response = engine.generate(
                messages=messages,
                tools=tools,
                temperature=temperature,
            )
            
            messages.append({"role": "assistant", "content": response})
            
            # 解析工具调用
            tool_call = parse_tool_call(response)
            
            if tool_call is None:
                # 没有工具调用，结束
                break
            
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})
            
            if tool_name == "localization_finish":
                # 完成工具调用
                result["called_finish_tool"] = True
                structured_locations = tool_args.get("locations", [])
                result["structured_locations"] = structured_locations
                break
            
            elif tool_name == "terminal":
                # 执行终端命令
                command = tool_args.get("command", "")
                output = execute_terminal(command, str(working_dir))
                messages.append({"role": "tool", "content": output})
            
            else:
                # 未知工具
                messages.append({"role": "tool", "content": f"Unknown tool: {tool_name}"})
        
        # 计算 reward
        total_reward = 0.0
        reward_details = {}
        
        for reward_fn_args in exp_config.reward:
            try:
                input_args = {
                    "final_message": response if 'response' in dir() else "",
                    "messages": messages,
                    "instance": instance,
                    "structured_locations": structured_locations,
                    **reward_fn_args.get("args", {})
                }
                
                reward_fn = get_reward_function(reward_fn_args["fn"])
                reward_outputs = reward_fn(**input_args)
                
                if isinstance(reward_outputs, tuple):
                    reward_value, reward_items = reward_outputs
                else:
                    reward_value = reward_outputs
                    reward_items = {reward_fn_args["fn"]: reward_value}
                
                weight = reward_fn_args.get("weight", 1.0)
                total_reward += reward_value * weight
                reward_details.update(reward_items)
                
            except Exception as e:
                reward_details[reward_fn_args["fn"]] = 0.0
        
        result["reward"] = total_reward
        result["reward_details"] = reward_details
        
    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["error"] += "\n" + traceback.format_exc()
    
    finally:
        # Cleanup
        try:
            if workspace.exists():
                os.system(f"rm -rf {str(workspace)}")
        except:
            pass
        
        result["wall_clock_duration"] = time.time() - start_time
    
    return result


def rollout_batch_sequential(
    instances: List[Dict],
    engine: OfflineVLLMEngine,
    system_prompt: str,
    user_prompt_template: str,
    tools: List[Dict],
    temperature: float,
    max_turns: int,
    exp_config: DictConfig,
    output_dir: str,
    resume: bool = False,
    n_samples: int = 1,
) -> List[Dict]:
    """顺序批量 rollout（单 vLLM 实例）
    
    Args:
        n_samples: 每个 instance 的 rollout 次数，取最高 reward 的结果
    """
    
    results = []
    
    # Check for existing results if resuming
    completed_ids = set()
    rollout_path = os.path.join(output_dir, "rollout_results.jsonl")
    
    if resume and os.path.exists(rollout_path):
        with open(rollout_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed_ids.add(r["instance_id"])
                    results.append(r)
                except:
                    pass
        print(f"Resuming: found {len(completed_ids)} completed instances")
    
    # Filter out completed instances
    pending_instances = [inst for inst in instances if inst["instance_id"] not in completed_ids]
    print(f"Pending instances: {len(pending_instances)}")
    
    if not pending_instances:
        return results
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sequential rollout
    with open(rollout_path, "a") as f:
        for inst in tqdm(pending_instances, desc="Rollout"):
            # N-samples: 对每个 instance 做 N 次 rollout，取最高 reward
            best_result = None
            for sample_idx in range(n_samples):
                result = rollout_single_instance(
                    inst,
                    engine,
                    system_prompt,
                    user_prompt_template,
                    tools,
                    temperature,
                    max_turns,
                    exp_config,
                )
                result["sample_idx"] = sample_idx
                
                if best_result is None or result.get("reward", 0) > best_result.get("reward", 0):
                    best_result = result
                
                # 如果已经得到高分（>= 3.0，即三级 F1 都接近满分），提前退出
                if result.get("reward", 0) >= 3.0:
                    break
            
            results.append(best_result)
            
            # Save incrementally
            f.write(json.dumps(best_result) + "\n")
            f.flush()
            
            # Print progress
            reward = best_result["reward"]
            called_tool = "✓" if best_result["called_finish_tool"] else "✗"
            error = "ERROR" if best_result["error"] else ""
            samples_tried = best_result.get("sample_idx", 0) + 1
            print(f"  {best_result['instance_id']}: reward={reward:.3f} tool={called_tool} samples={samples_tried}/{n_samples} {error}")
    
    return results


def _worker_rollout(
    worker_id: int,
    instance_queue: mp.Queue,
    result_queue: mp.Queue,
    model_path: str,
    tp_size: int,
    gpu_memory_utilization: float,
    device_ids: List[int],
    system_prompt: str,
    user_prompt_template: str,
    tools: List[Dict],
    temperature: float,
    max_turns: int,
    exp_config_dict: Dict,
    n_samples: int,
):
    """Worker 进程：加载模型并处理 instance 队列"""
    
    # 设置 CUDA_VISIBLE_DEVICES
    cuda_devices = ",".join(str(d) for d in device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    
    print(f"[Worker {worker_id}] Starting with GPUs: {cuda_devices}")
    
    # 加载模型
    engine = OfflineVLLMEngine(
        model_path=model_path,
        tp_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        engine_id=worker_id,
    )
    
    # 重建 exp_config
    exp_config = OmegaConf.create(exp_config_dict)
    
    # 处理队列中的 instance
    while True:
        try:
            item = instance_queue.get(timeout=1)
            if item is None:  # 结束信号
                break
            
            inst = item
            
            # N-samples rollout
            best_result = None
            for sample_idx in range(n_samples):
                result = rollout_single_instance(
                    inst,
                    engine,
                    system_prompt,
                    user_prompt_template,
                    tools,
                    temperature,
                    max_turns,
                    exp_config,
                )
                result["sample_idx"] = sample_idx
                result["worker_id"] = worker_id
                
                if best_result is None or result.get("reward", 0) > best_result.get("reward", 0):
                    best_result = result
                
                if result.get("reward", 0) >= 3.0:
                    break
            
            result_queue.put(best_result)
            
        except Exception as e:
            if "Empty" not in str(type(e).__name__):
                print(f"[Worker {worker_id}] Error: {e}")
            continue
    
    print(f"[Worker {worker_id}] Finished")


def rollout_batch_parallel(
    instances: List[Dict],
    model_path: str,
    tp_size: int,
    dp_size: int,
    gpu_memory_utilization: float,
    system_prompt: str,
    user_prompt_template: str,
    tools: List[Dict],
    temperature: float,
    max_turns: int,
    exp_config: DictConfig,
    output_dir: str,
    resume: bool = False,
    n_samples: int = 1,
    visible_gpus: Optional[List[int]] = None,
) -> List[Dict]:
    """数据并行批量 rollout（多 vLLM 实例）
    
    Args:
        dp_size: 数据并行度，启动多少个 vLLM 实例
        visible_gpus: 可用的 GPU 列表，默认 [0,1,...,7]
    """
    
    results = []
    
    # Check for existing results if resuming
    completed_ids = set()
    rollout_path = os.path.join(output_dir, "rollout_results.jsonl")
    
    if resume and os.path.exists(rollout_path):
        with open(rollout_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed_ids.add(r["instance_id"])
                    results.append(r)
                except:
                    pass
        print(f"Resuming: found {len(completed_ids)} completed instances")
    
    # Filter out completed instances
    pending_instances = [inst for inst in instances if inst["instance_id"] not in completed_ids]
    print(f"Pending instances: {len(pending_instances)}")
    
    if not pending_instances:
        return results
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 分配 GPU
    if visible_gpus is None:
        visible_gpus = list(range(8))  # 默认 8 个 GPU
    
    total_gpus = len(visible_gpus)
    gpus_per_worker = tp_size
    actual_dp_size = min(dp_size, total_gpus // gpus_per_worker)
    
    if actual_dp_size < dp_size:
        print(f"Warning: Requested dp_size={dp_size} but only have {total_gpus} GPUs with tp_size={tp_size}")
        print(f"Using dp_size={actual_dp_size}")
    
    print(f"\n{'='*60}")
    print(f"Data Parallel Configuration")
    print(f"{'='*60}")
    print(f"Total GPUs: {total_gpus}")
    print(f"Tensor parallel (TP): {tp_size}")
    print(f"Data parallel (DP): {actual_dp_size}")
    print(f"N-samples per instance: {n_samples}")
    print(f"{'='*60}\n")
    
    # 创建队列
    instance_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 将 instance 放入队列
    for inst in pending_instances:
        instance_queue.put(inst)
    
    # 添加结束信号
    for _ in range(actual_dp_size):
        instance_queue.put(None)
    
    # 分配 GPU 给每个 worker
    worker_gpu_assignments = []
    for i in range(actual_dp_size):
        start_idx = i * gpus_per_worker
        worker_gpus = visible_gpus[start_idx:start_idx + gpus_per_worker]
        worker_gpu_assignments.append(worker_gpus)
        print(f"Worker {i}: GPUs {worker_gpus}")
    
    # 转换 exp_config 为 dict（用于跨进程传递）
    exp_config_dict = OmegaConf.to_container(exp_config, resolve=True)
    
    # 启动 worker 进程
    workers = []
    for i in range(actual_dp_size):
        p = mp.Process(
            target=_worker_rollout,
            args=(
                i,
                instance_queue,
                result_queue,
                model_path,
                tp_size,
                gpu_memory_utilization,
                worker_gpu_assignments[i],
                system_prompt,
                user_prompt_template,
                tools,
                temperature,
                max_turns,
                exp_config_dict,
                n_samples,
            ),
        )
        p.start()
        workers.append(p)
    
    # 收集结果
    collected = 0
    total_pending = len(pending_instances)
    
    with open(rollout_path, "a") as f:
        pbar = tqdm(total=total_pending, desc="Rollout (DP)")
        while collected < total_pending:
            try:
                result = result_queue.get(timeout=300)  # 5 分钟超时
                results.append(result)
                collected += 1
                
                # Save incrementally
                f.write(json.dumps(result) + "\n")
                f.flush()
                
                # Print progress
                reward = result["reward"]
                called_tool = "✓" if result["called_finish_tool"] else "✗"
                error = "ERROR" if result["error"] else ""
                worker_id = result.get("worker_id", "?")
                samples_tried = result.get("sample_idx", 0) + 1
                pbar.set_postfix({"last": f"W{worker_id} r={reward:.2f}"})
                pbar.update(1)
                
            except Exception as e:
                print(f"Warning: Queue timeout or error: {e}")
                # 检查 worker 是否还活着
                alive_workers = sum(1 for p in workers if p.is_alive())
                if alive_workers == 0:
                    print("All workers have terminated!")
                    break
        
        pbar.close()
    
    # 等待所有 worker 结束
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    
    return results


# ========== 过滤和保存 ==========

def filter_by_reward(results: List[Dict], min_reward: float) -> List[Dict]:
    """按 reward 过滤"""
    filtered = [r for r in results if r.get("reward", 0) > min_reward and r.get("error") is None]
    print(f"Filtered: {len(filtered)}/{len(results)} samples with reward > {min_reward}")
    return filtered


def sort_by_difficulty(results: List[Dict]) -> List[Dict]:
    """按难度排序（高 reward = 简单，放前面）"""
    return sorted(results, key=lambda x: x.get("reward", 0), reverse=True)


def load_data(input_path: str) -> pd.DataFrame:
    """加载数据"""
    print(f"Loading data from {input_path}")
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    elif input_path.endswith(".jsonl"):
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    print(f"Loaded {len(df)} samples")
    return df


def save_filtered_dataset(
    results: List[Dict],
    original_df: pd.DataFrame,
    output_dir: str,
    split_ratio: float = 0.95
) -> Dict:
    """保存过滤后的数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get instance IDs in order
    filtered_ids = [r["instance_id"] for r in results]
    reward_map = {r["instance_id"]: r["reward"] for r in results}
    
    # Filter original dataframe
    filtered_df = original_df[original_df["instance_id"].isin(filtered_ids)].copy()
    filtered_df["prefilter_reward"] = filtered_df["instance_id"].map(reward_map)
    filtered_df = filtered_df.sort_values("prefilter_reward", ascending=False)
    
    # Split train/val
    n_train = int(len(filtered_df) * split_ratio)
    train_df = filtered_df.iloc[:n_train]
    val_df = filtered_df.iloc[n_train:]
    
    # Save
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    print(f"Saved {len(train_df)} train samples to {train_path}")
    print(f"Saved {len(val_df)} val samples to {val_path}")
    
    # Statistics
    # 注意：multilevel_localization_f1_reward 范围是 0-3（三级 F1），加 format_reward 最大可达 4
    rewards = [r.get("reward", 0) for r in results]
    max_reward = max(rewards) if rewards else 0
    
    stats = {
        "total_rollouts": len(results),
        "filtered_samples": len(filtered_df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "called_finish_tool": len([r for r in results if r.get("called_finish_tool")]),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
        "max_reward": max_reward,
        "min_reward": min(rewards) if rewards else 0,
        "reward_distribution": {
            "0": len([r for r in rewards if r == 0]),
            "0-1": len([r for r in rewards if 0 < r <= 1]),
            "1-2": len([r for r in rewards if 1 < r <= 2]),
            "2-3": len([r for r in rewards if 2 < r <= 3]),
            "3-4": len([r for r in rewards if 3 < r <= 4]),
            ">4": len([r for r in rewards if r > 4]),
        }
    }
    
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")
    
    return stats


# ========== Main ==========

def main():
    args = parse_args()
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    if args.filter_only:
        # 从已有 rollout 结果过滤
        if not args.rollout_dir:
            raise ValueError("--rollout-dir is required when using --filter-only")
        
        rollout_path = os.path.join(args.rollout_dir, "rollout_results.jsonl")
        print(f"Loading rollout results from {rollout_path}")
        results = []
        with open(rollout_path) as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except:
                    pass
        print(f"Loaded {len(results)} rollout results")
        
        df = load_data(args.input if args.input else find_dataset_path())
        filtered = filter_by_reward(results, args.min_reward)
        sorted_results = sort_by_difficulty(filtered)
        
        if args.max_samples:
            sorted_results = sorted_results[:args.max_samples]
        
        if not args.dry_run:
            stats = save_filtered_dataset(sorted_results, df, args.output)
            print("\nStatistics:")
            print(json.dumps(stats, indent=2))
        else:
            print(f"[Dry run] Would save {len(sorted_results)} samples to {args.output}")
        
        return
    
    # 需要做 rollout
    if not args.model:
        raise ValueError("--model is required for rollout")
    
    model_config = MODEL_CONFIGS[args.model]
    tp_size = args.tp_size or model_config["tp_size"]
    
    # 确定模型路径（支持 checkpoint）
    if args.checkpoint:
        model_path = args.checkpoint
        model_name = f"checkpoint:{args.checkpoint}"
        print(f"Using checkpoint: {args.checkpoint}")
    else:
        model_path = model_config["model_path"]
        model_name = model_config["model_name"]
    
    # 计算默认 DP size
    if args.dp_size is None:
        # 默认使用所有 8 个 GPU
        dp_size = 8 // tp_size
    else:
        dp_size = args.dp_size
    
    # 解析可见 GPU
    if args.visible_gpus:
        visible_gpus = [int(g.strip()) for g in args.visible_gpus.split(",")]
    else:
        visible_gpus = list(range(8))
    
    # 加载配置
    exp_config = OmegaConf.load(args.config)
    system_prompt = load_system_prompt(args.config)
    user_prompt_template = load_user_prompt_template(args.config)
    tools = get_tool_definitions()
    
    # 加载数据
    input_path = args.input if args.input else find_dataset_path()
    df = load_data(input_path)
    
    # Shuffle 数据（推荐，避免数据分布偏差）
    if args.shuffle:
        print(f"Shuffling data with seed={args.seed}")
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    if args.max_samples:
        df = df.head(args.max_samples)
    instances = df.to_dict("records")
    
    print(f"\n{'='*60}")
    print(f"Prefilter Configuration")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Tensor parallel (TP): {tp_size}")
    print(f"Data parallel (DP): {dp_size}")
    print(f"N-samples per instance: {args.n_samples}")
    print(f"Visible GPUs: {visible_gpus}")
    print(f"Config: {args.config}")
    print(f"Temperature: {args.temperature}")
    print(f"Max turns: {args.max_turns}")
    print(f"Total instances: {len(instances)}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    # 选择 rollout 方式
    if dp_size > 1:
        # 数据并行 rollout
        print(f"Using Data Parallel rollout with {dp_size} workers")
        results = rollout_batch_parallel(
            instances,
            model_path=model_path,
            tp_size=tp_size,
            dp_size=dp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            tools=tools,
            temperature=args.temperature,
            max_turns=args.max_turns,
            exp_config=exp_config,
            output_dir=args.output,
            resume=args.resume,
            n_samples=args.n_samples,
            visible_gpus=visible_gpus,
        )
    else:
        # 单实例顺序 rollout
        print("Using Sequential rollout with single vLLM instance")
        engine = OfflineVLLMEngine(
            model_path=model_path,
            tp_size=tp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            device_ids=visible_gpus[:tp_size] if visible_gpus else None,
        )
        
        results = rollout_batch_sequential(
            instances,
            engine,
            system_prompt,
            user_prompt_template,
            tools,
            args.temperature,
            args.max_turns,
            exp_config,
            args.output,
            args.resume,
            args.n_samples,
        )
    
    # 统计
    total = len(results)
    success = len([r for r in results if r.get("error") is None])
    called_tool = len([r for r in results if r.get("called_finish_tool")])
    with_reward = len([r for r in results if r.get("reward", 0) > 0])
    avg_reward = sum(r.get("reward", 0) for r in results) / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Rollout Summary")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Success: {success} ({100*success/total:.1f}%)")
    print(f"Called localization_finish: {called_tool} ({100*called_tool/total:.1f}%)")
    print(f"With reward > 0: {with_reward} ({100*with_reward/total:.1f}%)")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"{'='*60}")
    
    if args.rollout_only:
        print(f"\nRollout results saved to {args.output}/rollout_results.jsonl")
        return
    
    # 过滤和保存
    filtered = filter_by_reward(results, args.min_reward)
    sorted_results = sort_by_difficulty(filtered)
    
    if not args.dry_run:
        stats = save_filtered_dataset(sorted_results, df, args.output)
        print("\nFinal Statistics:")
        print(json.dumps(stats, indent=2))
    else:
        print(f"\n[Dry run] Would save {len(sorted_results)} samples to {args.output}")


if __name__ == "__main__":
    # 使用 spawn 方法启动子进程，避免 CUDA 初始化问题
    mp.set_start_method("spawn", force=True)
    main()
