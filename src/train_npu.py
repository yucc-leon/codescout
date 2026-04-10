"""
NPU entry point for CodeScout training.
Imports the NPU patch first.
"""
# === NPU patch must be the very first import ===
import npu_support.patch_cuda  # noqa: F401, isort:skip

# Now import everything else normally
from src.train import main

if __name__ == "__main__":
    main()
