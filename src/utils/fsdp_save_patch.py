"""
Monkey patch for FSDP save_hf_model to support configurable precision.

This patch fixes the issue where FSDP exports models in fp32 format instead of
the training precision (typically bfloat16). The patch allows specifying the
target dtype for model export, defaulting to the model's original config dtype
or the FSDP mixed precision config.
"""

import os
import torch
from typing import Union, Optional
from torch import nn


def patch_fsdp_save_hf_model():
    """
    Monkey patch FSDPStrategy.save_hf_model to support configurable export precision.
    
    This patch modifies the save_hf_model method to:
    1. Accept an optional target_dtype parameter
    2. Auto-detect dtype from model config or FSDP mixed precision config
    3. Convert state dict to target dtype before saving
    """
    try:
        from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
        from skyrl_train.distributed.fsdp_utils import fsdp_version, fsdp2_get_full_state_dict, PrecisionType
        from skyrl_train.model_wrapper import HFModelWrapper
        from skyrl_train.utils.io import io
        import torch.distributed as dist
        
        original_save_hf_model = FSDPStrategy.save_hf_model
        
        def patched_save_hf_model(
            self, 
            model: Union[HFModelWrapper, nn.Module], 
            output_dir: str, 
            tokenizer=None,
            target_dtype: Optional[Union[str, torch.dtype]] = None,
            **kwargs
        ) -> None:
            """
            Save model in HuggingFace safetensors format with configurable precision.
            
            Args:
                model: The model to save
                output_dir: Directory to save the model
                tokenizer: Optional tokenizer to save
                target_dtype: Target dtype for export. Can be:
                    - torch.dtype (e.g., torch.bfloat16, torch.float16)
                    - str (e.g., "bfloat16", "float16", "fp32")
                    - None (auto-detect from model config or FSDP config)
                **kwargs: Additional arguments passed to save_pretrained
            """
            # Step 1: Create output directory (rank 0 only)
            if self.is_rank_0():
                io.makedirs(output_dir, exist_ok=True)
                self.print(f"[rank-0]: Created output directory: {output_dir}")

            # Step 2: Extract models
            model_to_save = self._unwrap_model(model)
            fsdp_model = model.model if isinstance(model, HFModelWrapper) else model

            # Validate HuggingFace model
            if not hasattr(model_to_save, "config") or not hasattr(model_to_save, "save_pretrained"):
                raise ValueError("Model must be a HuggingFace model with config and save_pretrained method")

            # Step 3: Determine target dtype — use same precision as base model (read from base path config).
            # Do not use model_to_save.config at save time: FSDP2 init/materialization can set it to fp32.
            if target_dtype is None:
                base_path = getattr(self.model_config, "path", None) if getattr(self, "model_config", None) else None
                if base_path and self.is_rank_0():
                    try:
                        from transformers import AutoConfig
                        base_config = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
                        raw = getattr(base_config, "torch_dtype", None) or getattr(base_config, "dtype", None)
                        if raw is not None:
                            target_dtype = PrecisionType.to_dtype(raw) if isinstance(raw, str) else raw
                            if target_dtype in (torch.float32, torch.float16, torch.bfloat16):
                                self.print(f"[rank-0]: Using dtype from base model config ({base_path}): {target_dtype}")
                    except Exception as e:
                        if self.is_rank_0():
                            self.print(f"[rank-0]: Could not read base model config from {base_path}: {e}, falling back")
                        target_dtype = None
                if target_dtype is None:
                    mixed_precision_config = self.fsdp_config.get("mixed_precision", None)
                    if mixed_precision_config is not None:
                        target_dtype = PrecisionType.to_dtype(
                            mixed_precision_config.get("param_dtype", "bf16")
                        )
                    else:
                        target_dtype = torch.bfloat16
                    if self.is_rank_0():
                        self.print(f"[rank-0]: Using dtype from FSDP config / default: {target_dtype}")
            elif isinstance(target_dtype, str):
                # Convert string to torch.dtype
                target_dtype = PrecisionType.to_dtype(target_dtype)
                if self.is_rank_0():
                    self.print(f"[rank-0]: Using specified dtype: {target_dtype}")

            # Normalize: config.torch_dtype can be str (e.g. "bfloat16"), v.to(dtype=...) requires torch.dtype
            if isinstance(target_dtype, str):
                target_dtype = PrecisionType.to_dtype(target_dtype)

            # Step 4: Collect full state dict
            fsdp_ver = fsdp_version(fsdp_model)
            self.print(f"[rank-{self.get_rank()}]: Detected FSDP version: {fsdp_ver}")

            if fsdp_ver == 2:
                output_state_dict = fsdp2_get_full_state_dict(
                    fsdp_model, cpu_offload=True, rank0_only=True
                )
            elif fsdp_ver == 1:
                from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

                options = StateDictOptions(
                    full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False
                )
                output_state_dict = get_model_state_dict(fsdp_model, options=options)
                if not self.is_rank_0():
                    output_state_dict.clear()
            else:
                raise ValueError(f"Unsupported FSDP version: {fsdp_ver}")

            # Step 5: Convert state dict to target dtype (rank 0 only)
            if self.is_rank_0() and output_state_dict:
                self.print("[SAVE_PRECISION_PATCH] patched save_hf_model running (rank-0)")
                # Count tensors by dtype before conversion
                dtype_counts = {}
                for v in output_state_dict.values():
                    if isinstance(v, torch.Tensor):
                        dtype_str = str(v.dtype)
                        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
                
                self.print(f"[rank-0]: State dict dtype distribution before conversion: {dtype_counts}")
                
                # Convert tensors to target dtype
                converted_count = 0
                for k, v in output_state_dict.items():
                    if isinstance(v, torch.Tensor) and v.dtype != target_dtype:
                        # Only convert floating point tensors
                        if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                            output_state_dict[k] = v.to(dtype=target_dtype)
                            converted_count += 1
                
                self.print(
                    f"[SAVE_PRECISION_PATCH] Converted {converted_count} tensors to {target_dtype} (export will be bf16)"
                )
                self.print(
                    f"[rank-0]: Converted {converted_count} tensors to {target_dtype}"
                )

            # Step 6: Save on rank 0 only
            if self.is_rank_0():
                # Log actual dtype we are about to write (so we can verify patch ran)
                if output_state_dict:
                    sample = next((v for v in output_state_dict.values() if isinstance(v, torch.Tensor)), None)
                    if sample is not None:
                        self.print(f"[SAVE_PRECISION_PATCH] About to save state_dict with dtype={sample.dtype}")
                with io.local_work_dir(output_dir) as work_dir:
                    # Write state_dict ourselves with safetensors so dtype is guaranteed (HF save_pretrained
                    # can ignore passed state_dict in some paths and re-export from model -> fp32).
                    from safetensors.torch import save_file as safetensors_save_file
                    weights_name = "model.safetensors"
                    weights_path = os.path.join(work_dir, weights_name)
                    safetensors_save_file(output_state_dict, weights_path)
                    self.print(f"[SAVE_PRECISION_PATCH] Wrote {weights_name} with dtype={target_dtype}")

                    # Fix and save the config (ensure exported config reflects target dtype)
                    config_to_save = self._fix_fsdp_config(model_to_save.config)
                    if hasattr(config_to_save, "torch_dtype"):
                        config_to_save.torch_dtype = target_dtype
                    if hasattr(config_to_save, "dtype"):
                        config_to_save.dtype = str(target_dtype).replace("torch.", "")
                    config_to_save.save_pretrained(work_dir)

                    # Save tokenizer if provided
                    if tokenizer is not None:
                        tokenizer.save_pretrained(work_dir)

                self.print(f"[rank-0]: Successfully saved model to {output_dir} with dtype {target_dtype}")

            dist.barrier()
        
        # Apply the patch
        FSDPStrategy.save_hf_model = patched_save_hf_model
        print("[PATCH] Successfully patched FSDPStrategy.save_hf_model for configurable precision export")
        
    except ImportError as e:
        print(f"[PATCH] Warning: Could not patch FSDPStrategy.save_hf_model: {e}")
    except Exception as e:
        print(f"[PATCH] Error while patching FSDPStrategy.save_hf_model: {e}")
        raise


if __name__ == "__main__":
    # Apply patch when module is imported
    patch_fsdp_save_hf_model()
