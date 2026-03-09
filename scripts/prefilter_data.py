#!/usr/bin/env python3
"""
Prefilter data by running lightweight offline rollouts.

Core features merged from acs_aditya_upstream:
- rollout + filter flow
- filter-only / rollout-only
- N-samples per instance (pick best reward)
- data-parallel rollout with multiple vLLM workers
- curriculum style sorting by reward (easy -> hard)

Usage:
  # Full pipeline (rollout + filter -> train/validation.parquet)
  python scripts/prefilter_data.py --model 4b --output out/prefilter_baseline

  # Rollout only (save to output/rollout_results.jsonl)
  python scripts/prefilter_data.py --model 4b --output out/rollout --rollout-only

  # Filter only (from existing rollout dir)
  python scripts/prefilter_data.py --filter-only --rollout-dir out/rollout --input data/.../train.parquet --output out/filtered

  # See scripts/README_prefilter_data.md for full options and examples.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


MODEL_CONFIGS = {
    "4b": {
        "model_path": os.environ.get("MODEL_PATH_4B", "Qwen/Qwen3-4B-Instruct-2507"),
        "model_name": "Qwen3-4B-Instruct-2507",
        "tp_size": 1,
    },
    "14b": {
        "model_path": os.environ.get("MODEL_PATH_14B", "Qwen/Qwen3-14B-Instruct-2507"),
        "model_name": "Qwen3-14B-Instruct-2507",
        "tp_size": 2,
    },
}

DATASET_PATHS = [
    "data/swe_smith/train.parquet",
    "data/adityasoni17__SWE-smith-py-code-search_train/train.parquet",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prefilter script for curriculum training")
    parser.add_argument("--model", type=str, choices=["4b", "14b"], help="Model size")
    parser.add_argument("--input", type=str, default=None, help="Input dataset path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rewards/baseline_4b.yaml",
        help="Experiment config path (must include reward/tools/prompts)",
    )
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--filter-only", action="store_true")
    parser.add_argument("--rollout-dir", type=str, help="Existing rollout dir for --filter-only")
    parser.add_argument("--min-reward", type=float, default=0.0)
    parser.add_argument("--max-reward", type=float, default=None,
                        help="Upper bound for reward filter (exclusive). E.g. 3.0 to exclude perfect scores")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-samples", type=int, default=1, help="Rollouts per instance, use mean reward")
    parser.add_argument("--dp-size", type=int, default=None, help="Data parallel workers")
    parser.add_argument("--visible-gpus", type=str, default=None, help="e.g. 0,1,2,3")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path as model")
    parser.add_argument("--target-valid-samples", type=int, default=None,
                        help="Stop rollout early once this many samples fall in (min_reward, max_reward)")
    return parser.parse_args()


def find_dataset_path() -> str:
    root = Path(__file__).parent.parent
    for rel in DATASET_PATHS:
        p = root / rel
        if p.exists():
            print(f"Found dataset at: {p}")
            return str(p)
    raise FileNotFoundError(f"Dataset not found. Tried: {DATASET_PATHS}")


class OfflineVLLMEngine:
    def __init__(
        self,
        model_path: str,
        tp_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 64000,
        device_ids: Optional[List[int]] = None,
        engine_id: int = 0,
    ):
        from vllm import LLM

        self.engine_id = engine_id
        if device_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
            print(f"[Engine {engine_id}] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

        rope_scaling = {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 32768,
        }
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            rope_scaling=rope_scaling,
            disable_log_stats=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print(f"[Engine {engine_id}] Model ready: {model_path}")

    def generate(self, messages: List[Dict], tools: List[Dict], temperature: float = 0.6, max_tokens: int = 4096) -> str:
        from vllm import SamplingParams

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=["<|im_end|>"])
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        return outputs[0].outputs[0].text


def get_tool_definitions() -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Execute bash command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "localization_finish",
                "description": "Submit final localization with locations list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "locations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string"},
                                    "class_name": {"type": "string"},
                                    "function_name": {"type": "string"},
                                },
                                "required": ["file"],
                            },
                        }
                    },
                    "required": ["locations"],
                },
            },
        },
    ]


def load_system_prompt(config_path: str) -> str:
    cfg = OmegaConf.load(config_path)
    p = Path(__file__).parent.parent / "src" / "prompts" / cfg.prompts.system_prompt
    with open(p, "r") as f:
        return f.read()


def load_user_prompt_template(config_path: str) -> str:
    cfg = OmegaConf.load(config_path)
    p = Path(__file__).parent.parent / "src" / "prompts" / cfg.prompts.user_prompt
    with open(p, "r") as f:
        return f.read()


def format_user_prompt(template: str, instance: Dict, working_dir: str) -> str:
    from jinja2 import Template

    return Template(template).render(instance=instance, working_dir=working_dir)


def parse_tool_call(response: str) -> Optional[Dict]:
    import re

    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def execute_terminal(command: str, working_dir: str, timeout: int = 30) -> str:
    import subprocess

    try:
        ret = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = ret.stdout or ""
        if ret.stderr:
            out += "\n" + ret.stderr
        return out[:4000]
    except subprocess.TimeoutExpired:
        return "[Command timed out]"
    except Exception as e:
        return f"[Error: {e}]"


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
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.rewards import get_reward_function
    from src.utils.instance import clone_instance

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

    workspace = Path(f"/tmp/testbed/{str(uuid.uuid4())[:8]}/")
    try:
        status, working_dir = clone_instance(repo_name, commit_id, instance_id, workspace, patch)
        if not status:
            result["error"] = f"Clone failed: {repo_name}"
            return result
    except Exception as e:
        result["error"] = f"Clone exception: {e}"
        return result

    t0 = time.time()
    try:
        user_prompt = format_user_prompt(user_prompt_template, instance, str(working_dir))
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        structured_locations = None
        response = ""

        for turn in range(max_turns):
            result["num_turns"] = turn + 1
            response = engine.generate(messages=messages, tools=tools, temperature=temperature)
            messages.append({"role": "assistant", "content": response})
            tc = parse_tool_call(response)
            if tc is None:
                break
            name = tc.get("name", "")
            args = tc.get("arguments", {}) or {}
            if name == "localization_finish":
                result["called_finish_tool"] = True
                structured_locations = args.get("locations", [])
                result["structured_locations"] = structured_locations
                break
            if name == "terminal":
                cmd = args.get("command", "")
                messages.append({"role": "tool", "content": execute_terminal(cmd, str(working_dir))})
            else:
                messages.append({"role": "tool", "content": f"Unknown tool: {name}"})

        total_reward = 0.0
        reward_details = {}
        for reward_fn_args in exp_config.reward:
            try:
                input_args = {
                    "final_message": response,
                    "messages": messages,
                    "instance": instance,
                    "structured_locations": structured_locations,
                    **reward_fn_args.get("args", {}),
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
            except Exception:
                reward_details[reward_fn_args["fn"]] = 0.0

        result["reward"] = total_reward
        result["reward_details"] = reward_details
    except Exception as e:
        import traceback

        result["error"] = str(e) + "\n" + traceback.format_exc()
    finally:
        try:
            if workspace.exists():
                os.system(f"rm -rf {workspace}")
        except Exception:
            pass
        result["wall_clock_duration"] = time.time() - t0

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
    min_reward: float = 0.0,
    max_reward: Optional[float] = None,
    target_valid_samples: Optional[int] = None,
) -> List[Dict]:
    results: List[Dict] = []
    os.makedirs(output_dir, exist_ok=True)
    rollout_path = os.path.join(output_dir, "rollout_results.jsonl")

    completed_ids = set()
    if resume and os.path.exists(rollout_path):
        with open(rollout_path, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed_ids.add(r["instance_id"])
                    results.append(r)
                except Exception:
                    pass
        print(f"Resuming: found {len(completed_ids)} completed instances")

    pending = [inst for inst in instances if inst["instance_id"] not in completed_ids]
    print(f"Pending instances: {len(pending)}")
    if not pending:
        return results

    with open(rollout_path, "a") as fout:
        valid_count = _count_valid(results, min_reward, max_reward) if target_valid_samples else 0
        for inst in tqdm(pending, desc="Rollout"):
            sample_rewards = []
            best = None
            for sample_idx in range(n_samples):
                cur = rollout_single_instance(
                    inst,
                    engine,
                    system_prompt,
                    user_prompt_template,
                    tools,
                    temperature,
                    max_turns,
                    exp_config,
                )
                cur["sample_idx"] = sample_idx
                sample_rewards.append(cur.get("reward", 0))
                # keep the trajectory with highest reward for metadata
                if best is None or cur.get("reward", 0) > best.get("reward", 0):
                    best = cur
            # use mean reward across all samples for difficulty estimation
            mean_reward = sum(sample_rewards) / len(sample_rewards)
            best["reward"] = mean_reward
            best["sample_rewards"] = sample_rewards
            results.append(best)
            fout.write(json.dumps(best) + "\n")
            fout.flush()
            # early-stop check
            if target_valid_samples:
                if mean_reward > min_reward and (max_reward is None or mean_reward < max_reward) and best.get("error") is None:
                    valid_count += 1
                if valid_count >= target_valid_samples:
                    print(f"\n[early-stop] Reached {valid_count} valid samples (target={target_valid_samples}), stopping rollout")
                    break
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
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
    engine = OfflineVLLMEngine(
        model_path=model_path,
        tp_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        engine_id=worker_id,
    )
    exp_config = OmegaConf.create(exp_config_dict)

    while True:
        try:
            item = instance_queue.get(timeout=1)
            if item is None:
                break
            inst = item
            sample_rewards = []
            best = None
            for sample_idx in range(n_samples):
                cur = rollout_single_instance(
                    inst,
                    engine,
                    system_prompt,
                    user_prompt_template,
                    tools,
                    temperature,
                    max_turns,
                    exp_config,
                )
                cur["sample_idx"] = sample_idx
                cur["worker_id"] = worker_id
                sample_rewards.append(cur.get("reward", 0))
                if best is None or cur.get("reward", 0) > best.get("reward", 0):
                    best = cur
            # use mean reward across all samples for difficulty estimation
            mean_reward = sum(sample_rewards) / len(sample_rewards)
            best["reward"] = mean_reward
            best["sample_rewards"] = sample_rewards
            result_queue.put(best)
        except Exception:
            continue


def _count_valid(results: List[Dict], min_reward: float, max_reward: Optional[float]) -> int:
    """Count results that fall in the open interval (min_reward, max_reward)."""
    count = 0
    for r in results:
        if r.get("error") is not None:
            continue
        reward = r.get("reward", 0)
        if reward > min_reward and (max_reward is None or reward < max_reward):
            count += 1
    return count


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
    min_reward: float = 0.0,
    max_reward: Optional[float] = None,
    target_valid_samples: Optional[int] = None,
) -> List[Dict]:
    results: List[Dict] = []
    os.makedirs(output_dir, exist_ok=True)
    rollout_path = os.path.join(output_dir, "rollout_results.jsonl")

    completed_ids = set()
    if resume and os.path.exists(rollout_path):
        with open(rollout_path, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed_ids.add(r["instance_id"])
                    results.append(r)
                except Exception:
                    pass
        print(f"Resuming: found {len(completed_ids)} completed instances")

    pending = [inst for inst in instances if inst["instance_id"] not in completed_ids]
    print(f"Pending instances: {len(pending)}")
    if not pending:
        return results

    if visible_gpus is None:
        visible_gpus = list(range(8))
    total_gpus = len(visible_gpus)
    gpus_per_worker = tp_size
    actual_dp = min(dp_size, max(1, total_gpus // gpus_per_worker))

    instance_queue = mp.Queue()
    result_queue = mp.Queue()
    for inst in pending:
        instance_queue.put(inst)
    for _ in range(actual_dp):
        instance_queue.put(None)

    worker_gpu_assignments = []
    for i in range(actual_dp):
        start = i * gpus_per_worker
        worker_gpu_assignments.append(visible_gpus[start : start + gpus_per_worker])

    workers = []
    exp_config_dict = OmegaConf.to_container(exp_config, resolve=True)
    for i in range(actual_dp):
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

    early_stop_triggered = False
    with open(rollout_path, "a") as fout, tqdm(total=len(pending), desc="Rollout (DP)") as pbar:
        collected = 0
        valid_count = _count_valid(results, min_reward, max_reward) if target_valid_samples else 0
        while collected < len(pending):
            try:
                r = result_queue.get(timeout=300)
                results.append(r)
                collected += 1
                fout.write(json.dumps(r) + "\n")
                fout.flush()
                pbar.update(1)
                # early-stop: enough valid samples collected
                if target_valid_samples:
                    reward = r.get("reward", 0)
                    if reward > min_reward and (max_reward is None or reward < max_reward) and r.get("error") is None:
                        valid_count += 1
                    if valid_count >= target_valid_samples:
                        print(f"\n[early-stop] Reached {valid_count} valid samples (target={target_valid_samples}), stopping rollout")
                        early_stop_triggered = True
                        break
            except Exception:
                alive = sum(1 for p in workers if p.is_alive())
                if alive == 0:
                    break

    if early_stop_triggered:
        # Early-stop means we stop consuming result_queue before workers finish all pending
        # tasks; terminate workers explicitly to avoid hanging on queue backpressure.
        print("[cleanup] Early-stop triggered; terminating rollout workers...")
        for p in workers:
            if p.is_alive():
                p.terminate()

    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
        if p.is_alive():
            p.kill()  # SIGKILL as last resort

    # Avoid queue feeder-thread join deadlocks when parent exits early.
    for q in (instance_queue, result_queue):
        try:
            q.close()
        except Exception:
            pass
        try:
            q.cancel_join_thread()
        except Exception:
            pass

    return results


def filter_by_reward(results: List[Dict], min_reward: float, max_reward: float = None) -> List[Dict]:
    filtered = [r for r in results if r.get("reward", 0) > min_reward and r.get("error") is None]
    if max_reward is not None:
        filtered = [r for r in filtered if r.get("reward", 0) < max_reward]
    label = f"reward > {min_reward}" + (f" and reward < {max_reward}" if max_reward is not None else "")
    print(f"Filtered: {len(filtered)}/{len(results)} with {label}")
    return filtered


def sort_by_difficulty(results: List[Dict]) -> List[Dict]:
    return sorted(results, key=lambda x: x.get("reward", 0), reverse=True)


def load_data(input_path: str) -> pd.DataFrame:
    if input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    if input_path.endswith(".jsonl"):
        return pd.read_json(input_path, lines=True)
    raise ValueError(f"Unsupported format: {input_path}")


def save_filtered_dataset(results: List[Dict], original_df: pd.DataFrame, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    filtered_ids = [r["instance_id"] for r in results]
    reward_map = {r["instance_id"]: r["reward"] for r in results}

    filtered_df = original_df[original_df["instance_id"].isin(filtered_ids)].copy()
    filtered_df["prefilter_reward"] = filtered_df["instance_id"].map(reward_map)
    filtered_df = filtered_df.sort_values("prefilter_reward", ascending=False)

    train_path = os.path.join(output_dir, "train.parquet")
    filtered_df.to_parquet(train_path)

    rewards = [r.get("reward", 0) for r in results]
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_rollouts": len(results),
        "filtered_samples": len(filtered_df),
        "train_samples": len(filtered_df),
        "called_finish_tool": len([r for r in results if r.get("called_finish_tool")]),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
        "max_reward": max(rewards) if rewards else 0,
        "min_reward": min(rewards) if rewards else 0,
    }
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent.parent))

    if args.filter_only:
        if not args.rollout_dir:
            raise ValueError("--rollout-dir required when --filter-only")
        rollout_path = os.path.join(args.rollout_dir, "rollout_results.jsonl")
        results = []
        with open(rollout_path, "r") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
        df = load_data(args.input if args.input else find_dataset_path())
        filtered = sort_by_difficulty(filter_by_reward(results, args.min_reward, args.max_reward))
        if args.max_samples:
            filtered = filtered[: args.max_samples]
        if args.dry_run:
            print(f"[Dry run] Would save {len(filtered)} to {args.output}")
        else:
            stats = save_filtered_dataset(filtered, df, args.output)
            print(json.dumps(stats, indent=2))
        return

    if not args.model:
        raise ValueError("--model required for rollout")

    model_cfg = MODEL_CONFIGS[args.model]
    tp_size = args.tp_size or model_cfg["tp_size"]
    model_path = args.checkpoint if args.checkpoint else model_cfg["model_path"]

    # Fix 3: auto-resolve skyrl checkpoint dir to exported HF model
    # skyrl ckpt dirs lack config.json; the HF model lives under exported_model/global_step_*/policy/
    if model_path and os.path.isdir(model_path) and not os.path.isfile(os.path.join(model_path, "config.json")):
        exported_pattern = os.path.join(model_path, "exported_model", "global_step_*", "policy")
        candidates = sorted(glob.glob(exported_pattern), key=os.path.getmtime)
        if candidates:
            resolved = candidates[-1]  # latest by mtime
            print(f"[auto-resolve] skyrl ckpt -> HF model: {resolved}")
            model_path = resolved
        else:
            print(f"WARNING: {model_path} has no config.json and no exported_model found — vLLM may fail")

    if args.dp_size is None:
        dp_size = max(1, 8 // tp_size)
    else:
        dp_size = args.dp_size

    visible_gpus = [int(x.strip()) for x in args.visible_gpus.split(",")] if args.visible_gpus else list(range(8))

    exp_config = OmegaConf.load(args.config)
    system_prompt = load_system_prompt(args.config)
    user_prompt_template = load_user_prompt_template(args.config)
    tools = get_tool_definitions()

    input_path = args.input if args.input else find_dataset_path()
    df_full = load_data(input_path)
    df = df_full.copy()
    if args.shuffle:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if args.max_samples:
        df = df.head(args.max_samples)
    instances = df.to_dict("records")

    print("=" * 60)
    print("Prefilter Configuration")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"TP: {tp_size}, DP: {dp_size}, N-samples: {args.n_samples}")
    print(f"Config: {args.config}")
    print(f"Instances: {len(instances)}, Output: {args.output}")
    print(f"Reward filter: ({args.min_reward}, {args.max_reward})")
    if args.target_valid_samples:
        print(f"Target valid samples: {args.target_valid_samples} (early-stop)")
    print("=" * 60)

    if dp_size > 1:
        results = rollout_batch_parallel(
            instances=instances,
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
            min_reward=args.min_reward,
            max_reward=args.max_reward,
            target_valid_samples=args.target_valid_samples,
        )
    else:
        engine = OfflineVLLMEngine(
            model_path=model_path,
            tp_size=tp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            device_ids=visible_gpus[:tp_size] if visible_gpus else None,
        )
        results = rollout_batch_sequential(
            instances=instances,
            engine=engine,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            tools=tools,
            temperature=args.temperature,
            max_turns=args.max_turns,
            exp_config=exp_config,
            output_dir=args.output,
            resume=args.resume,
            n_samples=args.n_samples,
            min_reward=args.min_reward,
            max_reward=args.max_reward,
            target_valid_samples=args.target_valid_samples,
        )

    print("=" * 60)
    print("Rollout Summary")
    print("=" * 60)
    total = len(results)
    success = len([r for r in results if r.get("error") is None])
    called_tool = len([r for r in results if r.get("called_finish_tool")])
    with_reward = len([r for r in results if r.get("reward", 0) > 0])
    avg_reward = sum(r.get("reward", 0) for r in results) / total if total else 0
    print(f"Total: {total}, Success: {success}, Called finish: {called_tool}, Reward>0: {with_reward}")
    print(f"Average reward: {avg_reward:.4f}")
    print("=" * 60)

    if args.rollout_only:
        print(f"Rollout results saved to: {args.output}/rollout_results.jsonl")
        return

    filtered = sort_by_difficulty(filter_by_reward(results, args.min_reward, args.max_reward))
    if args.dry_run:
        print(f"[Dry run] Would save {len(filtered)} to {args.output}")
    else:
        stats = save_filtered_dataset(filtered, df_full, args.output)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        raise
    finally:
        # NOTE:
        # We occasionally hit a multiprocessing shutdown hang after early-stop
        # (summary printed, parquet/stats written, but parent process does not exit).
        # Force-exit here to bypass atexit/join deadlocks in mp/resource_tracker.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(exit_code)
