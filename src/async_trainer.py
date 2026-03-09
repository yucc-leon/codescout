import json
import os
import pathlib
import pickle
import re
import fsspec
import gcsfs
import numpy as np
import torch

from pathlib import Path
from loguru import logger
from typing import List, Set, Tuple

from skyrl_train.utils import ppo_utils, trainer_utils

from skyrl_train.generators.utils import get_rollout_metrics
from skyrl_train.generators.base import GeneratorOutput
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer, GeneratedOutputGroup
from omegaconf import OmegaConf


def _copy_wandb_history_to_new_run(
    entity: str,
    project: str,
    run_id: str,
    global_step: int,
    trainer,
) -> None:
    """从旧 run 拉取 step < global_step 的 history，复制到当前新 run。"""
    import wandb
    api = wandb.Api()
    run_path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    try:
        old_run = api.run(run_path)
    except Exception as e:
        logger.warning("Could not fetch old run %s for history copy: %s", run_path, e)
        return
    try:
        rows = []
        for row in old_run.scan_history(page_size=500):
            step = row.get("_step")
            if step is not None and step < global_step:
                rows.append(row)
        if not rows:
            logger.info("No history rows to copy (step < %s).", global_step)
            return
        rows.sort(key=lambda r: r.get("_step", 0))
        skip_keys = {"_step", "_timestamp", "_runtime", "_wandb"}
        for row in rows:
            step = row.get("_step")
            if step is None:
                continue
            metrics = {
                k: v for k, v in row.items()
                if k not in skip_keys and not k.startswith("_")
                and v is not None and (not isinstance(v, float) or not np.isnan(v))
            }
            if metrics:
                wandb.log(metrics, step=int(step))
        logger.info("Copied %s history rows (step 0..%s) to new run.", len(rows), global_step - 1)
    except Exception as e:
        logger.warning("Failed to copy wandb history: %s", e, exc_info=True)


def _rewind_wandb_to_step_if_needed(trainer, global_step: int) -> None:
    """
    续训时若 checkpoint 的 global_step 小于当前 wandb run 的 step，
    会导致 log(step=global_step) 被 wandb 忽略。优先尝试 resume_from 回退；
    失败则复制旧 run 的 step 之前数据到新 run。
    """
    if global_step <= 0:
        return
    tracker = getattr(trainer, "tracker", None)
    if not tracker or not getattr(tracker, "logger", None) or "wandb" not in tracker.logger:
        return
    import wandb
    if wandb.run is None or getattr(wandb.run, "step", None) is None:
        return
    if global_step > wandb.run.step:
        return
    run_id = getattr(tracker, "wandb_run_id", None) or getattr(tracker, "_wandb_run_id", None)
    if not run_id:
        logger.warning(
            "Checkpoint global_step (%s) <= wandb run step (%s); starting new wandb run.",
            global_step, wandb.run.step,
        )
        wandb.finish()
        wandb.init(
            project=trainer.cfg.trainer.project_name,
            name=trainer.cfg.trainer.run_name + "_resumed",
            config=OmegaConf.to_container(trainer.cfg, resolve=True),
        )
        tracker.logger["wandb"] = wandb
        if hasattr(tracker, "_wandb_run_id"):
            tracker._wandb_run_id = wandb.run.id
        return
    prev_run_step = wandb.run.step
    entity = getattr(wandb.run, "entity", None)
    project = getattr(wandb.run, "project", None) or trainer.cfg.trainer.project_name
    logger.info(
        "Checkpoint step %s < wandb run step %s; attempting rewind to overwrite curve from step %s.",
        global_step, prev_run_step, global_step,
    )
    wandb.finish()
    try:
        wandb.init(
            project=trainer.cfg.trainer.project_name,
            name=trainer.cfg.trainer.run_name,
            config=OmegaConf.to_container(trainer.cfg, resolve=True),
            resume_from=f"{run_id}?_step={global_step}",
        )
        tracker.logger["wandb"] = wandb
        if hasattr(tracker, "_wandb_run_id"):
            tracker._wandb_run_id = wandb.run.id
        logger.info("Wandb rewind succeeded; curve will continue from step %s.", global_step)
    except Exception as e:
        logger.warning(
            "Wandb rewind failed (%s). Copying old run history (step < %s) into a new run.",
            e, global_step,
        )
        wandb.init(
            project=trainer.cfg.trainer.project_name,
            name=trainer.cfg.trainer.run_name,
            config=OmegaConf.to_container(trainer.cfg, resolve=True),
        )
        tracker.logger["wandb"] = wandb
        new_run_id = wandb.run.id
        if hasattr(tracker, "_wandb_run_id"):
            tracker._wandb_run_id = new_run_id
        _copy_wandb_history_to_new_run(entity, project, run_id, global_step, trainer)
        if trainer.cfg.trainer.get("ckpt_path"):
            wandb_id_file = os.path.join(trainer.cfg.trainer.ckpt_path, "wandb_run_id.txt")
            try:
                with open(wandb_id_file, "w") as f:
                    f.write(new_run_id)
                logger.info("Saved new wandb run id to %s for future resume.", wandb_id_file)
            except OSError as err:
                logger.warning("Could not write wandb_run_id.txt: %s", err)


def patched_concatenate_generator_outputs(generator_outputs: List[GeneratorOutput]) -> GeneratorOutput:
    """
    Concatenate the generator outputs of multiple batches.

    We only aggregate rollout metrics the can deduced by responses and rewards, but not
    those that use `env_metrics` or `env_classes`.
    """
    assert len(generator_outputs) > 0
    has_rollout_logprobs = [output.get("rollout_logprobs") is not None for output in generator_outputs]
    if any(has_rollout_logprobs) and not all(has_rollout_logprobs):
        raise ValueError(
            "generator outputs are expected to all have null rollout_logprobs or all non-null, but received a mix"
        )
    result: GeneratorOutput = {
        "prompt_token_ids": sum([output["prompt_token_ids"] for output in generator_outputs], []),
        "response_ids": sum([output["response_ids"] for output in generator_outputs], []),
        "rewards": sum([output["rewards"] for output in generator_outputs], []),
        "loss_masks": sum([output["loss_masks"] for output in generator_outputs], []),
        "stop_reasons": (
            sum([output["stop_reasons"] for output in generator_outputs], [])
            if "stop_reasons" in generator_outputs[0] and generator_outputs[0]["stop_reasons"] is not None
            else None
        ),
        "rollout_logprobs": (
            sum([output["rollout_logprobs"] for output in generator_outputs], [])
            if generator_outputs[0]["rollout_logprobs"] is not None
            else None
        ),
        "trajectory_ids": sum([output["trajectory_ids"] for output in generator_outputs], []),
        "is_last_step": sum([output["is_last_step"] for output in generator_outputs], []),
        "sample_trajectory_path": generator_outputs[0].get("sample_trajectory_path") if generator_outputs else None,
    }

    # propagate additional keys with list values as-is
    additional_keys = [
        key for key in generator_outputs[0] if key not in result and isinstance(generator_outputs[0][key], (int, float))
    ]
    additional_result = {}
    if len(additional_keys):
        logger.info(f"Attempting to concatenate values for additional keys {additional_keys}")
    for key in additional_keys:
        try:
            additional_result[key] = np.mean([generator_output[key] for generator_output in generator_outputs]).item()
        except Exception as e:
            logger.error(f"Error in aggregating key {key}: {e}", exc_info=True)

    # Re-aggregate rollout metrics
    rollout_metrics = get_rollout_metrics(result["response_ids"], result["rewards"])
    result["rollout_metrics"] = {**rollout_metrics, **additional_result}

    # Validate the generator output using the number of prompts
    # Import here to avoid circular dependency.
    from skyrl_train.utils.trainer_utils import validate_generator_output

    num_prompts = len(result["prompt_token_ids"])
    validate_generator_output(num_prompts, result)

    return result

class CustomFullyAsyncRayPPOTrainer(FullyAsyncRayPPOTrainer):

    def load_checkpoints(self) -> Tuple[int, str, Set[str]]:
        """Load checkpoint and rewind wandb to checkpoint step when behind, so resumed logs are not ignored."""
        global_step, checkpoint_path, consumed_data_uids_set = super().load_checkpoints()
        _rewind_wandb_to_step_if_needed(self, global_step)
        return global_step, checkpoint_path, consumed_data_uids_set

    def convert_generation_group_mini_batch_to_training_input(
        self, cur_generation_group_mini_batch: List[GeneratedOutputGroup]
    ) -> TrainingInputBatch:
        """Given a mini-batch of generated groups, concatenate them into a single GeneratorOutput, then convert to a TrainingInputBatch."""
        generator_outputs = []
        uids = []
        stalenesses = []
        staleness_violation_count = 0
        group_size = len(cur_generation_group_mini_batch[0].generator_output["response_ids"])
        for cur_generated_output_group in cur_generation_group_mini_batch:
            cur_staleness = self.global_step - cur_generated_output_group.global_step_when_scheduled
            stalenesses.append(cur_staleness)
            generator_outputs.append(cur_generated_output_group.generator_output)
            uids.extend([cur_generated_output_group.uid] * group_size)

            # Check staleness violation.
            if cur_staleness > self.max_staleness_steps:
                # TODO(Charlie): should we drop, drop and resample, or just log?
                logger.warning(
                    "Staleness control violated despite using AsyncStalenessManager: "
                    f"cur_staleness={cur_staleness}, max_staleness_steps={self.max_staleness_steps}.\n"
                    "If this happens too often, consider increasing max_staleness_steps, adjusting "
                    "trainer.fully_async.num_parallel_generation_workers, or adjusting generation-training GPU allocation.\n"
                    "See https://skyrl.readthedocs.io/en/latest/tutorials/fully_async.html#async-staleness-manager for more details."
                )
                staleness_violation_count += 1

        generator_output = patched_concatenate_generator_outputs(generator_outputs)
        assert generator_output["rollout_metrics"] is not None, "Rollout metrics should be non-null."
        self.all_metrics.update(generator_output["rollout_metrics"])

        # Log staleness statistics for this step
        self.all_metrics.update(
            {
                "async/staleness_mean": sum(stalenesses) / len(stalenesses),
                "async/staleness_max": max(stalenesses),
                "async/staleness_min": min(stalenesses),
                "async/staleness_ratio": sum(1 for s in stalenesses if s > 0) / len(stalenesses),
                "async/staleness_violation_count": staleness_violation_count,
            }
        )

        if getattr(self.cfg.trainer, "save_session_sample_every_step", False):
            self._save_one_session_sample(generator_output)

        # Convert rewards to per-token form and compute reward metrics before training conversion
        uids = generator_output["trajectory_ids"]
        step_wise_training = self.cfg.trainer.step_wise_training
        self.cfg.trainer.step_wise_training = False
        generator_output = self.postprocess_generator_output(generator_output, uids)

        # print example just for debugging
        vis = self.tokenizer.decode(generator_output["response_ids"][0])
        logger.info(f"Example generated: {vis}")
        
        # return self.convert_to_training_input(generator_output, uids)
        training_input = self.convert_to_training_input(generator_output, uids)
        self.cfg.trainer.step_wise_training = step_wise_training
        return training_input


    def _save_one_session_sample(self, generator_output: GeneratorOutput) -> None:
        """Sample trajectories and dump decoded sessions as JSON/TXT."""
        try:
            interval = getattr(self.cfg.trainer, "session_sample_interval", 1)
            if self.global_step % interval != 0:
                return

            out_dir = getattr(
                self.cfg.trainer, "session_sample_dir", None
            ) or os.path.join(self.cfg.trainer.ckpt_path, "session_samples")
            os.makedirs(out_dir, exist_ok=True)

            trajectory_ids = generator_output["trajectory_ids"]
            if not trajectory_ids:
                return

            num_instances = getattr(self.cfg.trainer, "session_sample_num_instances", 1)
            is_last_step = generator_output.get("is_last_step")
            step_wise_training = self.cfg.trainer.step_wise_training

            if step_wise_training and is_last_step is not None:
                instance_groups = {}
                for i, traj_id in enumerate(trajectory_ids):
                    if traj_id not in instance_groups:
                        instance_groups[traj_id] = []
                    instance_groups[traj_id].append(i)

                import random

                selected_instances = random.sample(
                    list(instance_groups.keys()),
                    min(num_instances, len(instance_groups)),
                )
            else:
                import random

                num_samples = len(trajectory_ids)
                selected_indices = random.sample(
                    range(num_samples),
                    min(num_instances, num_samples),
                )
                instance_groups = {i: [i] for i in selected_indices}
                selected_instances = list(instance_groups.keys())

            rewards = generator_output["rewards"]
            prompt_token_ids = generator_output["prompt_token_ids"]
            response_ids = generator_output["response_ids"]

            for instance_key in selected_instances:
                indices = instance_groups[instance_key]

                if step_wise_training:
                    traj_id_str = str(instance_key)
                else:
                    traj_id_str = trajectory_ids[instance_key]

                final_reward = None
                if is_last_step is not None:
                    for i in indices:
                        if is_last_step[i]:
                            r = rewards[i]
                            final_reward = r if isinstance(r, (int, float)) else (sum(r) if r else 0.0)
                            break
                if final_reward is None and indices:
                    r = rewards[indices[-1]]
                    final_reward = r if isinstance(r, (int, float)) else (sum(r) if r else 0.0)

                sample_path = generator_output.get("sample_trajectory_path")
                reward_dict = {}
                if sample_path and isinstance(sample_path, str) and os.path.isfile(sample_path):
                    try:
                        with open(sample_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        final_reward = data.get("total_reward", final_reward)
                        reward_dict = data.get("reward_dict", {})
                    except Exception:
                        pass

                turns_data = []
                raw_text_parts = []
                for turn_idx, i in enumerate(indices, 1):
                    prompt_ids = prompt_token_ids[i]
                    response_ids_i = response_ids[i]
                    full_token_ids = prompt_ids + response_ids_i

                    full_text = self.tokenizer.decode(full_token_ids, skip_special_tokens=False)
                    prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                    response_text = self.tokenizer.decode(response_ids_i, skip_special_tokens=False)

                    

                    turn_info = {
                        "turn": turn_idx,
                        "prompt_token_ids": prompt_ids,
                        "response_token_ids": response_ids_i,
                        "prompt_text": prompt_text,
                        "response_text": response_text,
                        "full_text": full_text,
                        "num_prompt_tokens": len(prompt_ids),
                        "num_response_tokens": len(response_ids_i),
                        "is_last_step": bool(is_last_step[i]) if is_last_step is not None else True,
                        "step_reward": rewards[i] if isinstance(rewards[i], (int, float)) else list(rewards[i]) if rewards[i] else 0.0,
                    }
                    turns_data.append(turn_info)

                    if turn_idx > 1:
                        raw_text_parts.append(f"\n{'='*80} TURN {turn_idx} {'='*80}\n")
                    raw_text_parts.append(full_text)

                warnings = []
                for turn in turns_data:
                    if turn["num_prompt_tokens"] < 10 or turn["num_response_tokens"] < 10:
                        warnings.append(
                            f"Turn {turn['turn']}: short sequence "
                            f"(prompt={turn['num_prompt_tokens']}, response={turn['num_response_tokens']})"
                        )

                if len(turns_data) > 1 and all(t["is_last_step"] for t in turns_data):
                    prompt_lens = [t["num_prompt_tokens"] for t in turns_data]
                    if len(set(prompt_lens)) <= 2:
                        warnings.append(
                            "All turns are marked as is_last_step=True with similar prompt lengths; "
                            "this can indicate grouped GRPO rollouts."
                        )

                json_output = {
                    "step": self.global_step,
                    "trajectory_id": traj_id_str,
                    "training_mode": "step_wise" if step_wise_training else "grpo",
                    "num_turns": len(turns_data),
                    "final_reward": final_reward,
                    "reward_dict": reward_dict,
                    "warnings": warnings,
                    "turns": turns_data,
                }

                json_path = os.path.join(out_dir, f"step_{self.global_step}_{traj_id_str}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_output, f, indent=2, ensure_ascii=False)

                txt_path = os.path.join(out_dir, f"step_{self.global_step}_{traj_id_str}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("".join(raw_text_parts))

                logger.info(f"Saved session sample for trajectory {traj_id_str}: {json_path}, {txt_path}")
        except Exception as e:
            logger.warning(f"Failed to save session sample: {e}", exc_info=True)

    def dump_data(self, data: TrainingInputBatch, file_name: str):
        """
        Dump data to pickle file
        """
        path = os.path.join(self.cfg.trainer.export_path, "dumped_data", f"{file_name}.pkl")
        # Check if traj_dir is a gcs path
        if path.startswith("gs://"):
            fs = gcsfs.GCSFileSystem()
        else:
            os.makedirs(pathlib.Path(path).parent, exist_ok=True)
            fs = fsspec.filesystem("file")

        # Save pkl file
        with fs.open(path, "wb") as f:
            pickle.dump(data, f)
