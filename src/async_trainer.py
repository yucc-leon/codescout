import os
import pathlib
import pickle
import fsspec
import gcsfs
import numpy as np
import torch

from pathlib import Path
from loguru import logger
from typing import List, Union
from collections import defaultdict

from skyrl_train.utils import ppo_utils, trainer_utils

from skyrl_train.generators.utils import get_rollout_metrics
from skyrl_train.generators.base import GeneratorOutput
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer, GeneratedOutputGroup


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
            # result[key] = sum([generator_output[key] for generator_output in generator_outputs], [])
            additional_result[key] = np.mean([generator_output[key] for generator_output in generator_outputs]).item()
        except Exception as e:
            logger.error(f"Error in aggregating key {key}: {e}", exc_info=True)

    # Re-aggregate rollout metrics
    rollout_metrics = get_rollout_metrics(result["response_ids"], result["rewards"])
    result["rollout_metrics"] = {**rollout_metrics, **additional_result}

    # Validate the generator output using the number of prompts
    # Import here to avoid circular dependency.
    from skyrl_train.utils.trainer_utils import validate_generator_output

    # print("trajectory_ids", result["trajectory_ids"])
    # print("rewards", result["rewards"])
    # print("is_last_step", result["is_last_step"])

    num_prompts = len(result["prompt_token_ids"])
    validate_generator_output(num_prompts, result)

    return result

class CustomFullyAsyncRayPPOTrainer(FullyAsyncRayPPOTrainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track unique samples consumed across training
        self._seen_trajectory_ids = set()
        self._total_rollouts = 0

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

        # Track unique samples consumed
        trajectory_ids = generator_output["trajectory_ids"]
        batch_unique_ids = set(trajectory_ids)
        new_unique_ids = batch_unique_ids - self._seen_trajectory_ids
        self._seen_trajectory_ids.update(batch_unique_ids)
        self._total_rollouts += len(trajectory_ids)
        
        self.all_metrics.update(
            {
                "progress/unique_samples_consumed": len(self._seen_trajectory_ids),
                "progress/total_rollouts": self._total_rollouts,
                "progress/batch_unique_samples": len(batch_unique_ids),
                "progress/batch_new_samples": len(new_unique_ids),
            }
        )

        # Convert rewards to per-token form and compute reward metrics before training conversion
        uids = generator_output["trajectory_ids"]
        step_wise_training = self.cfg.trainer.step_wise_training
        self.cfg.trainer.step_wise_training = False
        generator_output = self.postprocess_generator_output(generator_output, uids)

        # # Truncate prompt_token_ids to avoid OOM
        # max_prompt_len = self.cfg.trainer.max_prompt_length
        # if max_prompt_len == -1:
        #     pass
        # else:
        #     truncated_prompt_token_ids = []
        #     for prompt_ids in generator_output["prompt_token_ids"]:
        #         if len(prompt_ids) > max_prompt_len:
        #             truncated_prompt_token_ids.append(prompt_ids[-max_prompt_len:])
        #         else:
        #             truncated_prompt_token_ids.append(prompt_ids)
        #     generator_output["prompt_token_ids"] = truncated_prompt_token_ids

        # print example just for debugging
        vis = self.tokenizer.decode(generator_output["response_ids"][0])
        logger.info(f"Example generated: {vis}")
        
        # return self.convert_to_training_input(generator_output, uids)
        training_input = self.convert_to_training_input(generator_output, uids)
        self.cfg.trainer.step_wise_training = step_wise_training
        return training_input

    def postprocess_generator_output(self, generator_output: GeneratorOutput, uids: List[str]) -> GeneratorOutput:
        """
        Override skyrl's default postprocess to:
        1. Remove misleading pass@n (reward > 0 is meaningless for F1-based rewards)
        2. Log avg_raw_reward instead
        
        exact_match is computed per-trajectory in the generator (metrics/is_exact_match)
        and auto-aggregated as batch mean — no need to duplicate here.
        """
        generator_output_for_metrics = generator_output
        uids_for_metrics = uids
        if self.cfg.trainer.step_wise_training:
            generator_output_for_metrics = defaultdict(list)
            for key in generator_output:
                if isinstance(generator_output[key], list):
                    generator_output_for_metrics[key] = [
                        generator_output[key][i]
                        for i in range(len(generator_output[key]))
                        if generator_output["is_last_step"][i]
                    ]
            uids_for_metrics = [
                uid for uid, is_last in zip(uids, generator_output["is_last_step"]) if is_last
            ]

        # --- Compute reward stats ---
        rewards: Union[List[float], List[List[float]]] = generator_output_for_metrics["rewards"]
        if rewards and isinstance(rewards[0], list):
            mean_raw_reward = float(np.mean([sum(r) for r in rewards]))
        else:
            mean_raw_reward = float(np.mean(rewards)) if rewards else 0.0

        self.all_metrics.update({"reward/avg_raw_reward": mean_raw_reward})
        logger.info(f"reward/avg_raw_reward: {mean_raw_reward}")

        # --- Per-token reward conversion (same as parent) ---
        rewards_full: Union[List[float], List[List[float]]] = generator_output["rewards"]
        responses: List[List[int]] = generator_output["response_ids"]
        per_token_rewards: List[List[float]] = []
        if rewards_full and isinstance(rewards_full[0], list):
            per_token_rewards = rewards_full
        else:
            for reward, response in zip(rewards_full, responses):
                per_token_reward = [0.0] * len(response)
                per_token_reward[-1] = float(reward)
                per_token_rewards.append(per_token_reward)

        generator_output["rewards"] = per_token_rewards
        return generator_output

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
