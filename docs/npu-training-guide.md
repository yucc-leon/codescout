# CodeScout NPU (Ascend 910) 训练指南

## 硬件要求

| 项目 | 要求 |
|------|------|
| NPU | 8 × Ascend 910, 64GB HBM each |
| CANN | 8.3.RC1 (8.3.0.1.200) |
| OS | Ubuntu 22.04 (aarch64) |
| Driver | npu-smi 25.3.rc1 |

## 软件栈

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12 | openhands-sdk 需要 PEP 695 语法 |
| torch | 2.8.0+cpu | 必须用 `--no-deps` 安装 |
| torch_npu | 2.8.0.post2 | 必须用 `--no-deps` 安装 |
| vllm | 0.11.0 | `--no-deps` |
| vllm-ascend | 0.11.0rc1 (源码构建) | 见下方说明 |
| ray | 2.51.1 | |
| transformers | 4.57.6 | |
| SkyRL | codescout-npu 分支 | 基于 commit 81e5a97 + NPU patch |

完整依赖列表见 `requirements-npu.txt`。

## 快速开始

```bash
# 1. 创建 conda 环境
conda create -n codescout-npu python=3.12 -y
conda activate codescout-npu

# 2. 安装环境（自动化脚本）
bash codescout/scripts/setup_ascend_env.sh

# 3. 启动训练（SP=2, 8x8 配置）
bash codescout/scripts/run_async_training_npu_sp2.sh \
  -m /path/to/Qwen3-4B-Instruct-2507 \
  -d ./codescout/data/swe_smith \
  -s /path/to/ckpts \
  -r my-run-name
```

## 训练配置

推荐配置（已验证 reward 对齐 H200）：

| 参数 | 值 | 说明 |
|------|-----|------|
| sequence_parallel_size | 2 | 解决 64GB 显存下长序列 OOM |
| flash_attn | true | SP=2 依赖 FA2 路径 |
| use_sample_packing | true | SP=2 依赖 packing |
| train_batch_size | 8 | |
| n_samples_per_prompt | 8 | 每个 prompt 8 个 rollout |
| max_model_len | 40960 | vLLM 推理最大长度 |
| max_generate_length | 8192 | 单次生成最大长度 |
| temperature | 1.0 | |
| lr | 1e-6 | |
| policy_loss_type | gspo | |
| eps_clip_low/high | 0.0003/0.0004 | GSPO clip range |

## SkyRL NPU 适配

SkyRL 的 NPU 适配包含两部分：

### 1. Monkey-Patch (`npu_support/patch_cuda.py`)

通过 `.pth` 文件在 Python 启动时自动加载，做以下 patch：
- `torch.cuda` → `torch.npu` 模块级代理
- `init_device_mesh("cuda")` → `"npu"`
- `ray.remote(num_gpus=N)` → `resources={"NPU": N}`
- `init_process_group(backend="nccl")` → `"hccl"`
- `flash_attn` stub（提供 `unpad_input`/`pad_input` 的纯 PyTorch 实现）
- `torch.autocast(device_type="cuda")` → `"npu"`

安装：
```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cp -r SkyRL/npu_support $SITE/npu_support
cp SkyRL/npu_support/npu_autoload.pth $SITE/
```

### 2. 源码 Patch

9 个文件的 `cuda→npu`, `nccl→hccl`, `GPU→NPU` 替换：
```
skyrl-train/skyrl_train/distributed/fsdp_strategy.py
skyrl-train/skyrl_train/distributed/fsdp_utils.py
skyrl-train/skyrl_train/distributed/strategy.py
skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py
skyrl-train/skyrl_train/workers/worker.py
skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py
skyrl-train/skyrl_train/model_wrapper.py (flash_attn import 改为 try/except)
skyrl-train/skyrl_train/utils/utils.py (环境变量转发)
skyrl-train/skyrl_train/fully_async_trainer.py (weight sync 可选跳过)
```

可用自动化脚本应用：
```bash
python SkyRL/npu_support/apply_npu_patches.py SkyRL/skyrl-train/skyrl_train
```

或直接使用 `codescout-npu` 分支（已包含所有 patch）。

## vllm-ascend 源码构建

PyPI 上的 vllm-ascend 0.11.0rc1 的 build-requires 写死了 torch==2.7.1，
需要从源码构建以兼容 torch 2.8：

```bash
git clone --depth 1 --branch v0.11.0rc1 \
  https://github.com/vllm-project/vllm-ascend.git /tmp/vllm-ascend-src

# 修改版本约束
sed -i 's/"torch-npu==2.7.1"/"torch-npu==2.8.0.post2"/' /tmp/vllm-ascend-src/pyproject.toml
sed -i 's/"torch==2.7.1"/"torch==2.8.0"/' /tmp/vllm-ascend-src/pyproject.toml
sed -i 's/"numpy<2.0.0"/"numpy"/' /tmp/vllm-ascend-src/pyproject.toml
sed -i 's/VERSION_EQUAL "2.7.1"/VERSION_EQUAL "2.8.0"/' /tmp/vllm-ascend-src/CMakeLists.txt

# 构建
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pip install /tmp/vllm-ascend-src/ --no-deps --no-build-isolation
```

## site-packages 补丁

### transformers: continue_final_message 冲突修复

文件: `transformers/tokenization_utils_base.py`

```python
# 找到 (约第 1658 行):
        if continue_final_message:
            if add_generation_prompt:
                raise ValueError(
# 改为:
        if continue_final_message:
            add_generation_prompt = False
            if False:
                raise ValueError(
```

### openhands SDK: 双重编码 JSON 防御

文件: `openhands/sdk/agent/agent.py`

```python
# 找到 (约第 409 行):
            arguments = json.loads(tool_call.arguments)
# 在后面加:
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
```

## ⚠️ 重要：NPU 环境不能使用原项目的依赖安装方式

原项目使用 `uv sync` + `pyproject.toml` 安装依赖，这会拉取 CUDA 版本的 torch 和 SkyRL 上游代码，**在 NPU 环境下不可用**。

NPU 环境必须：
1. 手动安装 `torch==2.8.0` + `torch_npu==2.8.0.post2`（`--no-deps`）
2. 使用本 fork 的 SkyRL（`codescout-npu` 分支），`pip install -e` 安装
3. 从源码构建 vllm-ascend（PyPI 版本 pin 了 torch==2.7.1）
4. 所有 torch 相关包必须用 `--no-deps` 安装，否则 pip 会拉 CUDA 版本的 torch 覆盖 torch_npu

详见下方安装步骤或使用 `scripts/setup_ascend_env.sh` 一键安装。

## 环境搭建常见问题

### 1. vllm-ascend 构建失败：`python3: not found`

容器里可能没有 `python3` 命令。确保 conda 环境的 bin 目录在 PATH 中：
```bash
export PATH="$CONDA_ENV/bin:$PATH"
```

### 2. vllm-ascend 构建失败：`No module named 'pybind11'` / `'setuptools_scm'` / `'typing_extensions'`

这些是 vllm-ascend 的构建依赖，必须在构建前安装：
```bash
pip install numpy pybind11 setuptools_scm cmake typing_extensions "setuptools<76" pyyaml
```

### 3. git clone 通过 HTTP 代理失败：`Proxy CONNECT aborted`

Ubuntu 默认的 git 使用 GnuTLS，与某些 HTTP 代理不兼容。解决方案：
- 用 `curl` 下载 tarball 代替 `git clone`
- 或者升级 git 到使用 OpenSSL 的版本

### 4. pip 安装拉到错误的 transformers 版本

不锁版本的 `pip install` 可能拉到 transformers 5.x，与 vllm 0.11 不兼容。
使用 `requirements-npu.txt` 锁定所有依赖版本：
```bash
pip install -r requirements-npu.txt
```

### 5. torch 版本被覆盖

`pip install` 不加 `--no-deps` 会拉 CUDA 版本的 torch 覆盖 torch_npu。
所有 torch 相关包必须用 `--no-deps` 安装，并在 requirements 安装后重新确认。

### 1. 长序列显存限制

64GB HBM 下，`lm_head` 产生的 logits tensor `(B, S, V)` 在长序列时占用大量显存（glen > 34000 时约 11 GB），叠加 FSDP 参数和 activations 可能导致 OOM 或性能退化。

**解决方案**：使用 Ulysses Sequence Parallel（SP=2），每卡只处理 S/2 长度的序列。推荐训练脚本 `run_async_training_npu_sp2.sh` 已默认启用。

### 2. NPU 僵尸进程

训练中断后 NPU 上的 vLLM/Ray 进程可能不会自动清理。训练前建议运行：
```bash
ray stop --force
pkill -9 -f "vllm\|ray\|FSDP"
```

### 3. litellm 版本锁定

锁定 litellm==1.82.6，不要升级。

## 训练效果

SP=2 配置在 Ascend 910 (8×64GB) 上的 `multilevel_localization_f1_reward`
曲线与 H200 基本对齐（step 50 时约 1.7）。

每步耗时约 3-5 分钟（rollout + training），其中 rollout 占大部分时间。
