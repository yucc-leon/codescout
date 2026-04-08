# CodeScout 昇腾 NPU 适配恢复指南

> 最后更新: 2026-03-31
> 用于环境崩溃或代码丢失时快速恢复

## 1. 硬件与基础环境

| 项目 | 值 |
|------|-----|
| CANN | 8.3.RC1 (8.3.0.1.200) |
| NPU | 8 × Ascend 910, 64GB HBM each |
| OS | Ubuntu 22.04.5 LTS (aarch64) |
| conda | `/sharedata/liyuchen/miniforge3` |
| conda 环境 | `codescout-cann83` (Python 3.12.13) |

## 2. 已验证的软件栈

| 组件 | 版本 | 安装方式 |
|------|------|----------|
| torch | 2.8.0+cpu | `pip install torch==2.8.0 --no-deps` |
| torch_npu | 2.8.0.post2 | `pip install torch-npu==2.8.0.post2 --no-deps` |
| vllm | 0.11.0 | `pip install vllm==0.11.0 --no-deps` |
| vllm-ascend | 0.11.0rc2.dev (源码构建) | 见 Section 4 |
| ray | 2.51.1 | pip |
| transformers | 4.57.6 | pip |
| openhands-sdk | 1.7.1 (commit 85ecfd9) | 从 git 安装，见 Section 5 |
| openhands-tools | 1.7.1 | 同上 |
| openhands-workspace | 1.7.1 | 同上 |
| openhands-agent-server | 1.7.1 | 同上 |
| litellm | 1.82.6 (锁定，不要升级) | pip |
| skyrl-train | 0.2.0 (editable) | `pip install -e skyrl-train/ --no-deps` |
| skyrl-gym | 0.1.1 (editable) | `pip install -e skyrl-gym/ --no-deps` |

**关键：所有 torch 相关包必须用 `--no-deps` 安装，否则 pip 会拉 CUDA 版本的 torch 覆盖。**

## 3. Git 仓库状态

### SkyRL

- 分支: `codescout-npu` (commit 81e5a97)
- npu_support/ 目录从 `main` 分支 checkout
- 9 个文件的 NPU 源码 patch（cuda→npu, nccl→hccl, GPU→NPU）
- model_wrapper.py: eager→sdpa attention patch
- fully_async_trainer.py: SKYRL_SKIP_WEIGHT_SYNC 环境变量支持

### codescout

- 分支: `npu-ascend-adapt`
- scripts/run_async_training_npu.sh: NPU 训练脚本
- src/train_npu.py: NPU 训练入口
- src/agent/agent.py: `self._initialized = True` 修复

## 4. vllm-ascend 源码构建

PyPI 上的 vllm-ascend 0.11.0rc1 build-requires 写死了 torch==2.7.1，需要从源码构建：

```bash
git clone --depth 1 --branch v0.11.0rc1 https://github.com/vllm-project/vllm-ascend.git /tmp/vllm-ascend-src

# 1. 修改 pyproject.toml
sed -i 's/"torch-npu==2.7.1"/"torch-npu==2.8.0.post2"/' /tmp/vllm-ascend-src/pyproject.toml
sed -i 's/"torch==2.7.1"/"torch==2.8.0"/' /tmp/vllm-ascend-src/pyproject.toml
sed -i 's/"numpy<2.0.0"/"numpy"/' /tmp/vllm-ascend-src/pyproject.toml

# 2. 修改 CMakeLists.txt
sed -i 's/VERSION_EQUAL "2.7.1"/VERSION_EQUAL "2.8.0"/' /tmp/vllm-ascend-src/CMakeLists.txt
sed -i 's/"3.9" "3.10" "3.11"/"3.9" "3.10" "3.11" "3.12"/' /tmp/vllm-ascend-src/CMakeLists.txt
sed -i "s/import torch; print(torch.__version__)/import torch; print(torch.__version__.split('+')[0])/" /tmp/vllm-ascend-src/CMakeLists.txt

# 3. 修改 setup.py: 修复 python3 路径
#    把 torch_npu_command 里的 python3 替换为 conda env 的绝对路径
#    在 TORCH_NPU_PATH cmake arg 后添加:
#      cmake_args += [f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}"]

# 4. 构建
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pip install /tmp/vllm-ascend-src/ --no-deps --no-build-isolation
```

## 5. openhands SDK 安装

codescout pin 了特定 commit，不要用 PyPI 上的最新版：

```bash
COMMIT=85ecfd9333d2d2cc4404dd460fd38868d9b978e2
REPO=https://github.com/OpenHands/software-agent-sdk.git

pip install "openhands-sdk @ git+${REPO}@${COMMIT}#subdirectory=openhands-sdk" --no-deps
pip install "openhands-tools @ git+${REPO}@${COMMIT}#subdirectory=openhands-tools" --no-deps
pip install "openhands-workspace @ git+${REPO}@${COMMIT}#subdirectory=openhands-workspace" --no-deps
pip install "openhands-agent-server @ git+${REPO}@${COMMIT}#subdirectory=openhands-agent-server" --no-deps
```

## 6. site-packages 补丁（3 处）

这些补丁直接修改了 site-packages 里的文件，pip 重装会丢失。

### 6.1 transformers: continue_final_message 冲突修复

文件: `transformers/tokenization_utils_base.py`

vllm 0.11 在多轮对话时同时传 `continue_final_message=True` 和 `add_generation_prompt=True`，
transformers 4.57 会报错。修复：自动把 `add_generation_prompt` 设为 False。

```python
# 找到这段代码（约第 1658 行）:
        if continue_final_message:
            if add_generation_prompt:
                raise ValueError(

# 改为:
        if continue_final_message:
            add_generation_prompt = False  # patched: auto-fix conflict
            if False:
                raise ValueError(
```

### 6.2 openhands SDK: 双重编码 JSON 防御

文件: `openhands/sdk/agent/agent.py`

hermes tool parser 偶尔返回双重编码的 JSON arguments（str 而非 dict）。

```python
# 找到这段代码（约第 409 行）:
            arguments = json.loads(tool_call.arguments)

# 改为:
            arguments = json.loads(tool_call.arguments)
            # Defensive: handle double-encoded JSON from hermes parser
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
```

### 6.3 NPU monkey-patch 安装

```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cp -r SkyRL/npu_support $SITE/npu_support
cp SkyRL/npu_support/npu_autoload.pth $SITE/
```

## 7. SkyRL 源码 patch 清单

### 7.1 NPU 设备 patch（cuda→npu, nccl→hccl, GPU→NPU）

可用自动化脚本 `SkyRL/npu_support/apply_npu_patches.py` 应用，或手动修改以下文件：

```
skyrl-train/skyrl_train/distributed/fsdp_strategy.py    (6 处)
skyrl-train/skyrl_train/distributed/fsdp_utils.py       (9 处)
skyrl-train/skyrl_train/distributed/strategy.py          (7 处)
skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py     (3 处)
skyrl-train/skyrl_train/workers/worker.py                (2 处)
skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py (4 处)
skyrl-train/skyrl_train/utils/utils.py                   (代理环境变量转发)
```

### 7.2 SDPA attention patch

文件: `skyrl-train/skyrl_train/model_wrapper.py`

```python
# flash_attn import 改为 try/except:
try:
    from flash_attn.bert_padding import pad_input, unpad_input
except ImportError:
    pad_input = None
    unpad_input = None

# attn_implementation 默认值改为 sdpa:
# 原: self.attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
# 改: self.attn_implementation = "flash_attention_2" if use_flash_attention_2 else "sdpa"
```

**这个 patch 至关重要**：没有它，`flash_attn=false` 会用 eager attention（O(n²) 显存），
长序列必定 OOM。SDPA 是 O(n) 显存。

### 7.3 weight sync 可选跳过

文件: `skyrl-train/skyrl_train/fully_async_trainer.py`

添加了 `SKYRL_SKIP_WEIGHT_SYNC=1` 环境变量支持，用于调试时跳过 weight sync。
正常训练不需要设置此变量。

## 8. 训练脚本与原始 4B 脚本的差异

`codescout/scripts/run_async_training_npu.sh` vs `codescout/scripts/run_async_training_4b.sh`

| 参数 | NPU | H100/H200 | 原因 |
|------|-----|-----------|------|
| CANN setenv | `ascend-toolkit/latest/bin/setenv.bash` | N/A | 昇腾环境 |
| weight_sync_backend | hccl | nccl | 昇腾通信库 |
| flash_attn | false | true | NPU 无 flash_attn，用 SDPA |
| enforce_eager | true | false | NPU graph mode 不生效，实际无差别 |
| gpu_memory_utilization | 0.85 | 0.8 | 补偿 64GB vs 80GB |
| 代理设置 | 已禁用 | 可能有 | 代理不通会阻断 git clone |
| PYTORCH_NPU_ALLOC_CONF | expandable_segments:True | N/A | 减少显存碎片化 |
| 入口 | `src.train_npu` | `src.train` | NPU patch 需要先加载 |

其余参数（max_model_len=40960, batch=8, rollouts=8 等）完全一致。

## 9. 已知问题与规避

| 问题 | 影响 | 规避 |
|------|------|------|
| hermes tool parser JSON 解析失败 | ~4% rollout 的 tool call 不被识别 | 模型能力问题，RL 训练会改善 |
| context 超限（>40960 tokens） | 极少数多轮对话失败 | 可接受，不影响训练 |
| git clone TLS 错误 | 偶发 rollout 失败 | 已配置 `git config --global http.retry 3` |
| litellm 安全 | 供应链攻击风险 | 锁定 litellm==1.82.6，不要升级 |
| NPU 僵尸进程 | 训练中断后 NPU 不释放 | 训练前运行 `ray stop --force; pkill -9 -f "vllm\|ray\|FSDP"` |

## 10. 获取 NPU 适配代码

### 方案 A：直接使用 Fork（推荐快速上手）

```bash
git clone https://github.com/yucc-leon/SkyRL.git
cd SkyRL
git checkout codescout-npu  # commit 6e3ffa5, 基于 81e5a97 + NPU patch
pip install -e skyrl-train/ --no-deps --no-build-isolation
pip install -e skyrl-gym/ --no-deps --no-build-isolation
```

此 fork 包含所有 NPU 适配改动，checkout 后直接可用，无需额外 patch。

**注意：此 fork 仅在以下环境验证通过，其他版本组合未经测试：**

| 组件 | 已验证版本 |
|------|-----------|
| CANN | 8.3.RC1 |
| torch | 2.8.0 |
| torch_npu | 2.8.0.post2 |
| vllm | 0.11.0 |
| vllm-ascend | 0.11.0rc1 (源码构建) |
| Python | 3.12 |
| NPU | Ascend 910, 64GB HBM |

### 方案 B：官方代码 + Patch 文件（推荐可审计场景）

从 codescout 作者的 SkyRL fork 拉取指定 commit，然后应用 patch：

```bash
# 1. 获取 codescout 兼容的 base commit
git clone https://github.com/adityasoni9998/SkyRL.git
cd SkyRL
git checkout 81e5a97c7430503c0c4e6508497cc5aa01a0c624

# 2. 应用 NPU 适配 patch
git apply /path/to/codescout/patches/skyrl-ascend-npu.patch

# 3. 安装
pip install -e skyrl-train/ --no-deps --no-build-isolation
pip install -e skyrl-gym/ --no-deps --no-build-isolation
```

也可以从官方 NovaSky-AI/SkyRL 出发，详见 `codescout/docs/skyrl-fork-provenance.md` 方案 C。

Patch 文件位置：`codescout/patches/skyrl-ascend-npu.patch`
- 基于 commit: `81e5a97` (SkyRL codescout 兼容版本)
- 包含: 19 个文件, 1312 行新增, 53 行删除
- 内容: npu_support/ 目录 + 9 个源码文件的 device/attention/通信 patch

Patch 内容可通过 `git apply --stat` 预览，`git apply --check` 验证。

## 11. 快速恢复步骤

一键恢复脚本：

```bash
bash codescout/scripts/setup_ascend_env.sh
```

脚本会自动完成以下所有步骤：创建 conda 环境、安装依赖、构建 vllm-ascend、
应用 SkyRL NPU patch + SDPA patch、安装 openhands SDK、应用 site-packages 补丁、
安装 monkey-patch、运行验证。

如需手动恢复，参考以下步骤：

```bash
# 0. 前置条件：CANN 8.3.RC1 已安装，miniforge3 已安装

# 1. 创建 conda 环境
/sharedata/liyuchen/miniforge3/bin/conda create -n codescout-cann83 python=3.12 -y
PIP=/sharedata/liyuchen/miniforge3/envs/codescout-cann83/bin/pip

# 2. 安装 torch + torch_npu（必须 --no-deps）
$PIP install torch==2.8.0 --no-deps
$PIP install torch-npu==2.8.0.post2 --no-deps

# 3. 安装 vllm（--no-deps）
$PIP install vllm==0.11.0 --no-deps

# 4. 构建安装 vllm-ascend（见 Section 4）

# 5. 安装 vllm 运行时依赖
$PIP install numpy pyyaml msgspec pyzmq blake3 cbor2 cloudpickle compressed-tensors \
  depyf diskcache einops lark partial-json-parser prometheus-client \
  prometheus-fastapi-instrumentator psutil py-cpuinfo pybase64 setproctitle \
  scipy sentencepiece tiktoken openai-harmony gguf mistral_common uvloop uvicorn

# 6. 安装 SkyRL 训练依赖
$PIP install hydra-core==1.3.2 omegaconf accelerate peft wandb datasets \
  tensordict jaxtyping polars torchdata func_timeout loguru tqdm ninja \
  tensorboard "ray==2.51.1" hf_transfer gcsfs litellm==1.82.6 docker \
  tenacity authlib s3fs pybind11 "setuptools<76"

# 7. SkyRL: 切到旧版 + 应用 NPU patch
cd SkyRL
git checkout codescout-npu
git checkout main -- npu_support/  # 从 main 拿 npu_support
# 如果 stash 里有 NPU patch: git stash pop
# 否则用自动化脚本: python npu_support/apply_npu_patches.py skyrl-train/skyrl_train
# 手动应用 SDPA patch（见 Section 7.2）
$PIP install -e skyrl-train/ --no-deps --no-build-isolation
$PIP install -e skyrl-gym/ --no-deps --no-build-isolation

# 8. 安装 openhands SDK（见 Section 5）

# 9. 安装 openhands 运行时依赖
$PIP install agent-client-protocol deprecation "fakeredis[lua]" fastmcp lmnr \
  python-frontmatter sqlalchemy wsproto fastapi alembic python-json-logger \
  binaryornot aiosqlite libtmux bashlex tom-swe

# 10. 安装 NPU monkey-patch（见 Section 6.3）

# 11. 应用 site-packages 补丁（见 Section 6.1, 6.2）

# 12. git 重试配置
git config --global http.retry 3
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 30

# 13. 启动训练
cd /sharedata/liyuchen/workspace
export PATH=/sharedata/liyuchen/miniforge3/envs/codescout-cann83/bin:$PATH
bash codescout/scripts/run_async_training_npu.sh \
  -m /sharedata/liyuchen/models/Qwen3-4B-Instruct-2507 \
  -d ./codescout/data/swe_smith \
  -s /sharedata/liyuchen/ckpts/codescout-4b-cann83 \
  -r codescout-4b-cann83-8x8
```
