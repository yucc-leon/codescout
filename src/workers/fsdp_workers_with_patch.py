"""
FSDP Ray workers with save-precision patch applied when this module is loaded.

save_hf_model runs inside the policy/critic Ray worker process. Ray deserializes
the actor class by its __module__; if we only re-export the class from
skyrl_train.workers.fsdp.fsdp_worker, the worker loads that module and never
loads this one, so the patch never runs. We must define the Worker classes
*here* (as thin subclasses) and wrap with ray.remote so __module__ is this
module; then the worker loads this module first, runs the patch, then gets
the class. Result: exported_model is bf16.
"""

from src.utils.fsdp_save_patch import patch_fsdp_save_hf_model

patch_fsdp_save_hf_model()

import ray
from skyrl_train.workers.fsdp.fsdp_worker import (
    FSDPPolicyWorkerBase,
    FSDPCriticWorkerBase,
    FSDPRefWorkerBase,
)


class _PolicyWorker(FSDPPolicyWorkerBase):
    """Thin subclass so Ray deserialization loads this module (and runs the patch)."""
    pass


class _CriticWorker(FSDPCriticWorkerBase):
    pass


class _RefWorker(FSDPRefWorkerBase):
    pass


PolicyWorker = ray.remote(num_gpus=1)(_PolicyWorker)
CriticWorker = ray.remote(num_gpus=1)(_CriticWorker)
RefWorker = ray.remote(num_gpus=1)(_RefWorker)

__all__ = ["PolicyWorker", "CriticWorker", "RefWorker"]
