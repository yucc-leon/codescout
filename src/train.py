import hydra
import os
from omegaconf import DictConfig, OmegaConf, open_dict
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils.tracking import Tracking
from loguru import logger
import ray

import asyncio

# Apply FSDP save precision patch before any training code runs
from src.utils.fsdp_save_patch import patch_fsdp_save_hf_model
patch_fsdp_save_hf_model()

# Register custom advantage estimators (grpo_length_norm_sqrt, grpo_length_norm_linear, grpo_length_norm_log)
from src.utils.length_normalized_advantage import *  # noqa: F401, F403

from src.generator.code_search_generator import CodeSearchGenerator
from src.async_trainer import CustomFullyAsyncRayPPOTrainer as FullyAsyncRayPPOTrainer
# from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer


class WandbResumeTracking(Tracking):
    """Extended Tracking class that supports wandb resume."""

    def __init__(self, project_name, experiment_name, backends, config, wandb_run_id=None):
        # Don't call super().__init__ directly, we need to customize wandb init
        if isinstance(backends, str):
            backends = [backends]
        for backend in backends:
            assert backend in self.supported_backends, f"{backend} is not supported"

        self.logger = {}
        self._curriculum_step_offset = int(os.environ.get("CURRICULUM_STEP_OFFSET", "0"))

        if "wandb" in backends:
            import wandb
            from omegaconf import OmegaConf

            wandb_settings = wandb.Settings(init_timeout=300)

            # Support resume with existing run id
            if wandb_run_id:
                logger.info(f"Attempting to resume wandb run: {wandb_run_id}")
                try:
                    wandb.init(
                        project=project_name,
                        name=experiment_name,
                        config=OmegaConf.to_container(config, resolve=True),
                        id=wandb_run_id,
                        resume="allow",
                        settings=wandb_settings,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to resume wandb run {wandb_run_id}: {e}. "
                        "Starting a fresh wandb run instead (checkpoint resume is unaffected)."
                    )
                    # Clear any cached run id to ensure a truly fresh run
                    os.environ.pop("WANDB_RUN_ID", None)
                    wandb.init(
                        project=project_name,
                        name=experiment_name,
                        config=OmegaConf.to_container(config, resolve=True),
                        settings=wandb_settings,
                    )
            else:
                wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=OmegaConf.to_container(config, resolve=True),
                    settings=wandb_settings,
                )
            self.logger["wandb"] = wandb

            # Add a continuous curriculum step for cross-stage/chunk visualization.
            # This keeps framework-local step behavior unchanged and only augments wandb logs.
            original_wandb_log = wandb.log

            def wandb_log_with_curriculum_step(data=None, *args, **kwargs):
                payload = data if isinstance(data, dict) else {}
                step = kwargs.get("step", None)
                if step is not None:
                    payload = dict(payload)
                    payload["curriculum_global_step"] = self._curriculum_step_offset + int(step)
                return original_wandb_log(payload if isinstance(data, dict) else data, *args, **kwargs)

            wandb.log = wandb_log_with_curriculum_step
            # Declare metric so it's easy to switch X-axis in wandb UI.
            wandb.define_metric("curriculum_global_step")

            # Save run id for future resume
            self._wandb_run_id = wandb.run.id
            logger.info(f"Wandb run id: {self._wandb_run_id}")

        # Initialize other backends using parent logic
        if "mlflow" in backends:
            from skyrl_train.utils.tracking import _MlflowLoggingAdapter
            self.logger["mlflow"] = _MlflowLoggingAdapter(project_name, experiment_name, config)

        if "swanlab" in backends:
            import swanlab
            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)
            swanlab.init(
                project_name=project_name,
                experiment_name=experiment_name,
                config=config,
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "tensorboard" in backends:
            from skyrl_train.utils.tracking import _TensorboardAdapter
            self.logger["tensorboard"] = _TensorboardAdapter()

        if "console" in backends:
            from skyrl_train.utils.tracking import ConsoleLogger
            self.console_logger = ConsoleLogger()
            self.logger["console"] = self.console_logger

    @property
    def wandb_run_id(self):
        return getattr(self, '_wandb_run_id', None)


class CodeSearchPPOExp(BasePPOExp):
    def _setup_trainer(self):
        """Override to use FSDP workers with save-precision patch (export bf16)."""
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        if self.cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from src.workers.fsdp_workers_with_patch import PolicyWorker, CriticWorker, RefWorker
        elif self.cfg.trainer.strategy == "megatron":
            from skyrl_train.workers.megatron.megatron_worker import PolicyWorker, CriticWorker, RefWorker
        else:
            raise ValueError(f"Unknown strategy type: {self.cfg.trainer.strategy}")

        tracker = self.get_tracker()
        inference_engine_client = self.get_inference_client()
        generator = self.get_generator(self.cfg, self.tokenizer, inference_engine_client)
        trainer = self.get_trainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=self.colocate_pg,
        )
        trainer.build_models(PolicyWorker, CriticWorker, RefWorker)
        return trainer

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        generator = CodeSearchGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=OmegaConf.create({"max_env_workers": 0}),
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=self.cfg.trainer.policy.model.path,
            step_wise=cfg.trainer.get("step_wise_training", False),
        )
        return generator

    def get_tracker(self):
        """Initializes the tracker with wandb resume support."""
        wandb_run_id = self.cfg.trainer.get("wandb_run_id", None)

        tracker = WandbResumeTracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=self.cfg.trainer.logger,
            config=self.cfg,
            wandb_run_id=wandb_run_id,
        )

        # Save wandb run id to checkpoint directory for future resume
        if tracker.wandb_run_id and self.cfg.trainer.ckpt_path:
            wandb_id_file = os.path.join(self.cfg.trainer.ckpt_path, "wandb_run_id.txt")
            os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)
            with open(wandb_id_file, "w") as f:
                f.write(tracker.wandb_run_id)
            logger.info(f"Saved wandb run id to {wandb_id_file}")

        return tracker

    def run(self):
        """Override to add wandb cleanup after training."""
        trainer = self._setup_trainer()
        logger.info(
            "========== Training start: resume_mode=%s, ckpt_path=%s (will load checkpoint if mode allows) ==========",
            self.cfg.trainer.resume_mode,
            self.cfg.trainer.ckpt_path,
        )
        asyncio.run(trainer.train())

        # Explicitly finish wandb to ensure proper cleanup
        if hasattr(trainer, 'tracker') and hasattr(trainer.tracker, 'logger'):
            if 'wandb' in trainer.tracker.logger:
                logger.info("Finishing wandb run...")
                trainer.tracker.logger['wandb'].finish()
                logger.info("Wandb run finished successfully")


class AsyncCodeSearchPPOExp(CodeSearchPPOExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def run(self):
        trainer = self._setup_trainer()
        asyncio.run(trainer.train())

        # Explicitly finish wandb to ensure proper cleanup
        if hasattr(trainer, 'tracker') and hasattr(trainer.tracker, 'logger'):
            if 'wandb' in trainer.tracker.logger:
                logger.info("Finishing wandb run...")
                trainer.tracker.logger['wandb'].finish()
                logger.info("Wandb run finished successfully")


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    if cfg.get("run_async_trainer", False):
        print("Running async trainer")
        exp = AsyncCodeSearchPPOExp(cfg)
    else:
        print("Running sync trainer")
        exp = CodeSearchPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    # check cfg.generator.exp_config if it exists or not
    if hasattr(cfg.generator, "exp_config"):
        # Open yaml file and print its contents
        with open(cfg.generator.exp_config, "r") as f:
            exp_cfg = OmegaConf.load(f)

        with open_dict(cfg):
            cfg.generator.reward = exp_cfg.reward
            cfg.generator.tools = exp_cfg.tools
            # Parse prompts if they exist in the exp config
            if hasattr(exp_cfg, "prompts"):
                cfg.generator.prompts = exp_cfg.prompts
    else:
        with open_dict(cfg):
            cfg.generator.reward = [
                {"fn": "multilevel_localization_f1_reward"},
            ]
            cfg.generator.tools = [
                "terminal",
            ]

    # Set default prompts if not specified
    if not hasattr(cfg.generator, "prompts"):
        with open_dict(cfg):
            cfg.generator.prompts = {
                "system_prompt": "templates/system_prompt.j2",
                "user_prompt": "templates/file_module_parallel_tools.j2"
            }

    # Initialize Ray with excludes to avoid "Package size exceeds 512MB" errors
    from skyrl_train.utils.utils import prepare_runtime_environment
    from skyrl_train.utils.ppo_utils import sync_registries

    env_vars = prepare_runtime_environment(cfg)
    ray.init(runtime_env={
        "env_vars": env_vars,
        "excludes": [
            ".git/",
            ".venv/",
            "__pycache__/",
            "*.pyc",
            ".pytest_cache/",
            "logs/",
            "output/",
            "outputs/",
            "*.log",
            "models/",
            "*.ckpt",
            "*.pth",
            "*.bin",
            "wandb/",
            "swanlog/",
            "uv.lock",
        ],
    })
    sync_registries()

    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
