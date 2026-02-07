import hydra
import os
from omegaconf import DictConfig, OmegaConf, open_dict
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.tracking import Tracking
from loguru import logger
import ray

import asyncio

# from src.tools import tool_exists
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

        if "wandb" in backends:
            import wandb
            from omegaconf import OmegaConf

            # Support resume with existing run id
            if wandb_run_id:
                logger.info(f"Resuming wandb run: {wandb_run_id}")
                wandb.init(
                    project=project_name, 
                    name=experiment_name, 
                    config=OmegaConf.to_container(config, resolve=True),
                    id=wandb_run_id,
                    resume="must"
                )
            else:
                wandb.init(
                    project=project_name, 
                    name=experiment_name, 
                    config=OmegaConf.to_container(config, resolve=True)
                )
            self.logger["wandb"] = wandb
            
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
                project=project_name,
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
        # Start the async training loop
        asyncio.run(trainer.train())


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

    # cfg.trainer.policy.deepspeed_config.zero_optimization.offload_param.device = "cpu"
    # cfg.trainer.policy.deepspeed_config.zero_optimization.offload_optimizer.device = "cpu"
    # cfg.trainer.policy.deepspeed_config.zero_optimization.zero_hpz_partition_size = 8

    print("cfg.trainer.policy.deepspeed_config")
    print(cfg.trainer.policy.deepspeed_config)

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

    # # Check if the tool exists in the registry
    # for tool in cfg.generator.tools:
    #     if not tool_exists(tool):
    #         raise ValueError(f"Tool {tool} does not exist in the registry")
    
    # Set default prompts if not specified
    if not hasattr(cfg.generator, "prompts"):
        with open_dict(cfg):
            cfg.generator.prompts = {
                "system_prompt": "templates/system_prompt.j2",
                "user_prompt": "templates/file_module_parallel_tools.j2"
            }
    
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
