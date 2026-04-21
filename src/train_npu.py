"""
NPU entry point for CodeScout training.
Imports the NPU patch first.
"""
# === NPU patch must be the very first import ===
import npu_support.patch_cuda  # noqa: F401, isort:skip
npu_support.patch_cuda.ensure_patched()  # retry if .pth auto-load failed

# Now import everything else normally
from src.train import main

if __name__ == "__main__":
    main()
