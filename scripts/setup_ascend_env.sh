#!/bin/bash
# ============================================================================
# CodeScout 昇腾 NPU 环境一键搭建脚本
#
# 用法:
#   bash codescout/scripts/setup_ascend_env.sh
#
# 前置条件:
#   - CANN 8.3.RC1 已安装
#   - miniforge3 已安装在 /sharedata/liyuchen/miniforge3
#   - SkyRL 仓库在 /sharedata/liyuchen/workspace/SkyRL (codescout-npu 分支)
#   - codescout 仓库在 /sharedata/liyuchen/workspace/codescout
#
# 此脚本会:
#   1. 创建 conda 环境 codescout-cann83
#   2. 安装所有 Python 依赖
#   3. 从源码构建 vllm-ascend
#   4. 安装 SkyRL (editable) 并应用 NPU patch
#   5. 安装 openhands SDK (pinned version)
#   6. 应用 site-packages 补丁
#   7. 安装 NPU monkey-patch
# ============================================================================

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/sharedata/liyuchen/miniforge3}"
ENV_NAME="${ENV_NAME:-codescout-cann83}"
WORKSPACE="${WORKSPACE:-/sharedata/liyuchen/workspace}"
SKYRL_ROOT="$WORKSPACE/SkyRL"
CODESCOUT_ROOT="$WORKSPACE/codescout"

CONDA="$CONDA_BASE/bin/conda"
PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"

log() { echo "$(date '+%H:%M:%S') [setup] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ============================================================================
# Step 0: Preflight checks
# ============================================================================
log "Preflight checks..."
[ -f /usr/local/Ascend/ascend-toolkit/latest/version.cfg ] || die "CANN not found"
[ -x "$CONDA" ] || die "conda not found at $CONDA"
[ -d "$SKYRL_ROOT/skyrl-train" ] || die "SkyRL not found at $SKYRL_ROOT"
[ -d "$CODESCOUT_ROOT/src" ] || die "codescout not found at $CODESCOUT_ROOT"

CANN_VER=$(grep "toolkit_running_version" /usr/local/Ascend/ascend-toolkit/latest/version.cfg | head -1 | grep -oP '\d+\.\d+')
log "CANN version: $CANN_VER"

# ============================================================================
# Step 1: Create conda environment
# ============================================================================
if [ -d "$CONDA_BASE/envs/$ENV_NAME" ]; then
    log "Conda env '$ENV_NAME' already exists, skipping creation"
else
    log "Creating conda env '$ENV_NAME' (Python 3.12)..."
    $CONDA create -n $ENV_NAME python=3.12 -y
fi

# ============================================================================
# Step 2: Install torch + torch_npu (--no-deps to avoid pulling CUDA torch)
# ============================================================================
log "Installing torch 2.8.0 + torch_npu 2.8.0.post2..."
$PIP install torch==2.8.0 --no-deps 2>&1 | tail -1
$PIP install torch-npu==2.8.0.post2 --no-deps 2>&1 | tail -1

# Verify
$PYTHON -c "import torch; import torch_npu; assert torch.npu.is_available(), 'NPU not available'; print(f'torch {torch.__version__}, torch_npu {torch_npu.__version__}, {torch.npu.device_count()} NPUs')"
log "torch + torch_npu OK"

# ============================================================================
# Step 3: Install vllm 0.11.0
# ============================================================================
log "Installing vllm 0.11.0..."
$PIP install vllm==0.11.0 --no-deps 2>&1 | tail -1

# ============================================================================
# Step 4: Build vllm-ascend from source
# ============================================================================
log "Building vllm-ascend from source..."
VLLM_ASCEND_SRC=/tmp/vllm-ascend-src-$$

if [ -d "$VLLM_ASCEND_SRC" ]; then rm -rf "$VLLM_ASCEND_SRC"; fi
git clone --depth 1 --branch v0.11.0rc1 https://github.com/vllm-project/vllm-ascend.git "$VLLM_ASCEND_SRC"

# Patch pyproject.toml
sed -i 's/"torch-npu==2.7.1"/"torch-npu==2.8.0.post2"/' "$VLLM_ASCEND_SRC/pyproject.toml"
sed -i 's/"torch==2.7.1"/"torch==2.8.0"/' "$VLLM_ASCEND_SRC/pyproject.toml"
sed -i 's/"numpy<2.0.0"/"numpy"/' "$VLLM_ASCEND_SRC/pyproject.toml"

# Patch CMakeLists.txt
sed -i 's/VERSION_EQUAL "2.7.1"/VERSION_EQUAL "2.8.0"/' "$VLLM_ASCEND_SRC/CMakeLists.txt"
sed -i 's/"3.9" "3.10" "3.11"/"3.9" "3.10" "3.11" "3.12"/' "$VLLM_ASCEND_SRC/CMakeLists.txt"
sed -i "s/import torch; print(torch.__version__)/import torch; print(torch.__version__.split('+')[0])/" "$VLLM_ASCEND_SRC/CMakeLists.txt"

# Patch setup.py: fix python3 path and add ASCEND_PYTHON_EXECUTABLE
sed -i "s|torch_npu_command = \"python3 -m pip show torch-npu|torch_npu_command = \"'$PYTHON' -m pip show torch-npu|" "$VLLM_ASCEND_SRC/setup.py"
sed -i '/cmake_args += \[f"-DTORCH_NPU_PATH/a\        cmake_args += [f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}"]' "$VLLM_ASCEND_SRC/setup.py"

# Install build deps
$PIP install setuptools setuptools-scm cmake decorator einops scipy pybind11 msgspec numba wheel 2>&1 | tail -1

# Build
TORCH_DEVICE_BACKEND_AUTOLOAD=0 $PIP install "$VLLM_ASCEND_SRC/" --no-deps --no-build-isolation 2>&1 | tail -3
rm -rf "$VLLM_ASCEND_SRC"
log "vllm-ascend built OK"

# ============================================================================
# Step 5: Install runtime dependencies
# ============================================================================
log "Installing vllm runtime deps..."
$PIP install numpy pyyaml msgspec pyzmq blake3 cbor2 cloudpickle compressed-tensors \
  depyf diskcache einops lark partial-json-parser prometheus-client \
  prometheus-fastapi-instrumentator psutil py-cpuinfo pybase64 setproctitle \
  scipy sentencepiece tiktoken openai-harmony gguf mistral_common uvloop uvicorn 2>&1 | tail -1

log "Installing SkyRL training deps..."
$PIP install hydra-core==1.3.2 omegaconf accelerate peft wandb datasets \
  tensordict jaxtyping polars torchdata func_timeout loguru tqdm ninja \
  tensorboard "ray==2.51.1" hf_transfer gcsfs litellm==1.82.6 docker \
  tenacity authlib s3fs pybind11 "setuptools<76" 2>&1 | tail -1

# ============================================================================
# Step 6: Install SkyRL (editable) + apply NPU patches
# ============================================================================
log "Installing SkyRL..."

# Check if SkyRL already has NPU patches (committed)
if git -C "$SKYRL_ROOT" log --oneline -1 2>/dev/null | grep -q "Ascend NPU"; then
    log "SkyRL already has NPU patches committed, skipping source patch step"
else
    # Ensure npu_support exists (checkout from main if needed)
    if [ ! -f "$SKYRL_ROOT/npu_support/patch_cuda.py" ]; then
    log "Checking out npu_support from main branch..."
    git -C "$SKYRL_ROOT" checkout main -- npu_support/
fi

# Apply NPU device patches (cuda→npu, nccl→hccl, GPU→NPU)
find "$SKYRL_ROOT/skyrl-train" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
$PYTHON "$SKYRL_ROOT/npu_support/apply_npu_patches.py" "$SKYRL_ROOT/skyrl-train/skyrl_train"

# Apply SDPA attention patch (critical: prevents O(n²) OOM)
$PYTHON -c "
fpath = '$SKYRL_ROOT/skyrl-train/skyrl_train/model_wrapper.py'
with open(fpath) as f: content = f.read()
changed = False

old1 = 'from flash_attn.bert_padding import pad_input, unpad_input'
new1 = '''try:
    from flash_attn.bert_padding import pad_input, unpad_input
except ImportError:
    pad_input = None
    unpad_input = None'''
if old1 in content:
    content = content.replace(old1, new1); changed = True

old2 = '\"flash_attention_2\" if use_flash_attention_2 else \"eager\"'
new2 = '\"flash_attention_2\" if use_flash_attention_2 else \"sdpa\"'
if old2 in content:
    content = content.replace(old2, new2); changed = True

if changed:
    with open(fpath, 'w') as f: f.write(content)
    print('  model_wrapper.py: SDPA patch applied')
else:
    print('  model_wrapper.py: already patched or pattern not found')
"
fi

# Apply utils.py proxy forwarding patch (needed regardless of commit state)
$PYTHON -c "
fpath = '$SKYRL_ROOT/skyrl-train/skyrl_train/utils/utils.py'
with open(fpath) as f: content = f.read()
marker = '# Forward proxy settings to Ray workers'
if marker not in content:
    old = '    return env_vars'
    new = '''    # Forward proxy settings to Ray workers (needed for wandb etc.)
    for proxy_var in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'no_proxy', 'NO_PROXY'):
        if os.environ.get(proxy_var):
            env_vars[proxy_var] = os.environ[proxy_var]

    return env_vars'''
    if old in content:
        content = content.replace(old, new, 1)
        with open(fpath, 'w') as f: f.write(content)
        print('  utils.py: proxy forwarding patch applied')
    else:
        print('  utils.py: pattern not found')
else:
    print('  utils.py: already patched')
"

$PIP install -e "$SKYRL_ROOT/skyrl-train/" --no-deps --no-build-isolation 2>&1 | tail -1
$PIP install -e "$SKYRL_ROOT/skyrl-gym/" --no-deps --no-build-isolation 2>&1 | tail -1
log "SkyRL installed OK"

# ============================================================================
# Step 7: Install openhands SDK (pinned version)
# ============================================================================
log "Installing openhands SDK (commit 85ecfd9)..."
OH_COMMIT=85ecfd9333d2d2cc4404dd460fd38868d9b978e2
OH_REPO=https://github.com/OpenHands/software-agent-sdk.git

$PIP install "openhands-sdk @ git+${OH_REPO}@${OH_COMMIT}#subdirectory=openhands-sdk" --no-deps 2>&1 | tail -1
$PIP install "openhands-tools @ git+${OH_REPO}@${OH_COMMIT}#subdirectory=openhands-tools" --no-deps 2>&1 | tail -1
$PIP install "openhands-workspace @ git+${OH_REPO}@${OH_COMMIT}#subdirectory=openhands-workspace" --no-deps 2>&1 | tail -1
$PIP install "openhands-agent-server @ git+${OH_REPO}@${OH_COMMIT}#subdirectory=openhands-agent-server" --no-deps 2>&1 | tail -1

log "Installing openhands runtime deps..."
$PIP install agent-client-protocol deprecation "fakeredis[lua]" fastmcp lmnr \
  python-frontmatter sqlalchemy wsproto fastapi alembic python-json-logger \
  binaryornot aiosqlite libtmux bashlex tom-swe 2>&1 | tail -1

log "openhands SDK installed OK"

# ============================================================================
# Step 8: Apply site-packages patches
# ============================================================================
log "Applying site-packages patches..."
SITE=$($PYTHON -c "import site; print(site.getsitepackages()[0])")

# 8.1 transformers: continue_final_message conflict fix
$PYTHON -c "
import transformers.tokenization_utils_base as m
fpath = m.__file__
with open(fpath) as f: content = f.read()
old = '        if continue_final_message:\n            if add_generation_prompt:\n                raise ValueError('
new = '        if continue_final_message:\n            add_generation_prompt = False  # patched: auto-fix conflict\n            if False:\n                raise ValueError('
if old in content:
    content = content.replace(old, new)
    with open(fpath, 'w') as f: f.write(content)
    print('  transformers: continue_final_message patch applied')
else:
    print('  transformers: already patched or pattern not found')
"

# 8.2 openhands SDK: double-encoded JSON defense
$PYTHON -c "
import openhands.sdk.agent.agent as m
fpath = m.__file__
with open(fpath) as f: content = f.read()
marker = '# Defensive: handle double-encoded JSON'
if marker not in content:
    old = '            arguments = json.loads(tool_call.arguments)'
    new = '''            arguments = json.loads(tool_call.arguments)
            # Defensive: handle double-encoded JSON from hermes parser
            if isinstance(arguments, str):
                arguments = json.loads(arguments)'''
    if old in content:
        content = content.replace(old, new, 1)
        with open(fpath, 'w') as f: f.write(content)
        print('  openhands SDK: double-encoded JSON patch applied')
    else:
        print('  openhands SDK: pattern not found, may need manual patch')
else:
    print('  openhands SDK: already patched')
"

# 8.3 Install NPU monkey-patch to site-packages
if [ -f "$SITE/npu_support/patch_cuda.py" ]; then
    log "NPU monkey-patch already installed"
else
    rm -rf "$SITE/npu_support" 2>/dev/null
    cp -r "$SKYRL_ROOT/npu_support" "$SITE/npu_support"
    cp "$SKYRL_ROOT/npu_support/npu_autoload.pth" "$SITE/"
    log "NPU monkey-patch installed"
fi

# Clear all .pyc caches for patched files
find "$SITE/transformers" -name "tokenization_utils_base.cpython*.pyc" -delete 2>/dev/null
find "$SITE/openhands" -name "agent.cpython*.pyc" -delete 2>/dev/null
find "$SITE/npu_support" -name "*.pyc" -delete 2>/dev/null

# ============================================================================
# Step 9: Git config for retry
# ============================================================================
git config --global http.retry 3
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 30

# ============================================================================
# Step 10: Verify
# ============================================================================
log "Running verification..."
$PYTHON -c "
import torch, torch_npu
assert torch.npu.is_available(), 'NPU not available'
assert torch.npu.device_count() > 0, 'No NPU devices'

# Verify monkey-patch
assert torch.cuda.is_available(), 'Monkey-patch not working (torch.cuda should proxy to npu)'
assert torch.cuda.device_count() == torch.npu.device_count(), 'Monkey-patch device count mismatch'

# Verify vllm
import vllm
assert vllm.__version__ == '0.11.0', f'Wrong vllm version: {vllm.__version__}'

# Verify vllm-ascend
import vllm_ascend

# Verify SkyRL
from skyrl_train.entrypoints.main_base import BasePPOExp

# Verify openhands
import openhands.sdk

# Verify SDPA patch
from skyrl_train.model_wrapper import HFModelWrapper
w = HFModelWrapper.__init__
import inspect
src = inspect.getsource(w)
assert 'sdpa' in src, 'SDPA patch not applied in model_wrapper.py'

print('All checks passed!')
print(f'  torch: {torch.__version__}')
print(f'  torch_npu: {torch_npu.__version__}')
print(f'  vllm: {vllm.__version__}')
print(f'  NPUs: {torch.npu.device_count()}')
"

log "============================================"
log "Setup complete! To start training:"
log ""
log "  export PATH=$CONDA_BASE/envs/$ENV_NAME/bin:\$PATH"
log "  cd $WORKSPACE"
log "  bash codescout/scripts/run_async_training_npu.sh \\"
log "    -m /sharedata/liyuchen/models/Qwen3-4B-Instruct-2507 \\"
log "    -d ./codescout/data/swe_smith \\"
log "    -s /sharedata/liyuchen/ckpts/codescout-4b-cann83 \\"
log "    -r codescout-4b-cann83-8x8"
log "============================================"
