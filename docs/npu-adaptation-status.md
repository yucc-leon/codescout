# CodeScout NPU (Ascend) 适配状态文档

> 最后更新: 2026-03-27

## 1. 目标

在昇腾 NPU (Ascend 910) 上跑通 CodeScout 的 RL 训练。
CodeScout 基于 SkyRL 框架，使用 vLLM 做推理、FSDP2 做训练、Ray 做分布式调度。

## 2. 版本兼容性矩阵

### 已验证能工作的组合（cann-recipes-train 的 code RL 例子）

| 组件 | 版本 |
|------|------|
| CANN | **8.3.rc1** |
| Python | 3.11 |
| torch | 2.8.0 |
| torch_npu | 2.8.0.post2 |
| vllm | 0.11.0 |
| vllm-ascend | 0.11.0rc1 |
| verl | v0.7.0.dev (commit c651b7b) |

### 我们的目标组合

| 组件 | 版本 | 说明 |
|------|------|------|
| CANN | **8.3** (需降级，当前机器是 8.5) | 8.5 下 vllm-ascend 0.11 不稳定 |
| Python | 3.12 | openhands-sdk 需要 PEP 695 语法 |
| torch | 2.8.0 | |
| torch_npu | 2.8.0.post2 | |
| vllm | 0.11.0 | SkyRL 旧版 pin 的版本 |
| vllm-ascend | 0.11.0rc1 | |
| SkyRL | commit 81e5a97 (codescout-npu 分支) | codescout 兼容的版本 |

### CANN 8.5 下的问题

| vllm 版本 | vllm-ascend 版本 | 状态 |
|-----------|-----------------|------|
| 0.11.0 | 0.11.0rc1 | 推理单测能过，但训练循环中 HTTP endpoint 返回 503 |
| 0.14.1 | 0.14.0rc1 | vllm-ascend 编译失败 (bgmv kernel ld.lld error) |
| 0.16.0rc2 | 0.14.0rc2 (v013 env) | 能 serve，但 Python 3.11 不兼容 openhands-sdk |

### 为什么选 CANN 8.3

1. cann-recipes-train 的 RL 训练例子在 CANN 8.3 上验证过
2. vllm 0.11 + vllm-ascend 0.11 是为 CANN 8.3 开发的
3. 我们在 CANN 8.5 上遇到的 503 问题很可能是版本不匹配导致的

## 3. 已完成的适配工作

### 3.1 NPU Monkey-Patch (`SkyRL/npu_support/patch_cuda.py`)

通过 `.pth` 文件在 Python 启动时自动加载，做以下 patch：

| Patch | 说明 |
|-------|------|
| `torch.cuda` → `torch.npu` | 模块级 proxy，让所有 `torch.cuda.*` 调用透明走 NPU |
| `Tensor.cuda()` → `Tensor.npu()` | 方法级 patch |
| `init_device_mesh("cuda")` → `"npu"` | FSDP2 DeviceMesh |
| `ray.remote(num_gpus=N)` → `resources={"NPU": N}` | Ray 资源调度 |
| `ActorClass.options(num_gpus=N)` → `resources={"NPU": N}` | Ray actor 选项 |
| `placement_group({"GPU": N})` → `{"NPU": N}` | Ray placement group |
| `init_process_group(backend="nccl")` → `"hccl"` | 分布式通信 |
| `new_group(backend="nccl")` → `"hccl"` | 分布式子组 |
| `flash_attn` stub | 注入假的 flash_attn 模块避免 import 报错 |

**安装方式：**
```bash
cp -r SkyRL/npu_support /path/to/site-packages/npu_support
cp SkyRL/npu_support/npu_autoload.pth /path/to/site-packages/
```

### 3.2 SkyRL 源码 Patch（旧版 commit 81e5a97）

monkey-patch 无法解决 C++ dispatch 层面的 `torch.device("cuda:0")` 问题，
以下文件需要直接修改（`cuda` → `npu`，`nccl` → `hccl`，`GPU` → `NPU`）：

```
skyrl-train/skyrl_train/distributed/fsdp_strategy.py    (6 处)
skyrl-train/skyrl_train/distributed/fsdp_utils.py       (9 处)
skyrl-train/skyrl_train/distributed/strategy.py          (7 处)
skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py     (3 处)
skyrl-train/skyrl_train/workers/worker.py                (2 处)
skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py (4 处)
skyrl-train/skyrl_train/model_wrapper.py                 (flash_attn import 改为 try/except)
skyrl-train/skyrl_train/utils/utils.py                   (代理环境变量转发到 Ray)
```

自动化 patch 脚本在 `/tmp/npu_device_patch.py`（需要重新生成）。

### 3.3 skyrl_train Import Shim

codescout 用 `from skyrl_train.xxx import yyy`，但旧版 SkyRL 的 `skyrl_train`
是 `skyrl-train/skyrl_train/` 下的真实包，直接 `pip install -e SkyRL/skyrl-train/`
即可，不需要 shim。

新版 SkyRL (main) 把 `skyrl_train` 重构成了 `skyrl.train.*` + `skyrl.backends.skyrl_train.*`，
需要 shim 包（`SkyRL/skyrl_train/`）做映射。**如果用旧版 SkyRL 则不需要 shim。**

### 3.4 codescout 代码修复

| 文件 | 修复 | 原因 |
|------|------|------|
| `src/agent/agent.py` | 加 `self._initialized = True` | openhands-sdk 1.14.0 新增了 `_initialized` 检查 |
| `src/train_npu.py` | NPU 入口，import patch 后调 main | 确保 patch 在所有 import 之前加载 |

### 3.5 训练脚本 (`scripts/run_async_training_npu.sh`)

关键 NPU 特定参数：
```bash
# Ascend 环境变量
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
HCCL_CONNECT_TIMEOUT=360
HCCL_EXEC_TIMEOUT=360
HCCL_IF_BASE_PORT=64033
HCCL_OP_EXPANSION_MODE=AIV
VLLM_ASCEND_ENABLE_NZ=0

# 训练参数差异（vs CUDA 版本）
trainer.flash_attn=false          # NPU 用 SDPA 替代 flash_attn
generator.enforce_eager=true      # 避免 graph capture 问题
generator.weight_sync_backend=hccl  # HCCL 替代 NCCL
```

## 4. 已验证通过的环节

| 环节 | 状态 | 说明 |
|------|------|------|
| torch_npu tensor 运算 | ✅ | NPU 上 forward/backward/optimizer step 全通 |
| FSDP2 模型加载 + wrap | ✅ | Qwen3-4B bf16 + cpu_offload |
| FSDP2 forward + backward + optimizer | ✅ | 单卡验证通过 |
| Ray NPU 资源调度 | ✅ | placement_group 用 NPU 资源 |
| Ray worker 初始化 | ✅ | PolicyWorker, RefWorker, InferenceEngine 全部启动 |
| vLLM 单独推理 | ✅ | `LLM.generate()` 离线推理成功输出文本 |
| Async trainer 循环 | ✅ | generation buffer → policy train → step progress 全通 |
| Agent 初始化 | ✅ | `_initialized` 修复后不再报错 |
| Weight sync (HCCL broadcast) | ⚠️ | broadcast 本身完成，但之后 vLLM HTTP 503 |
| vLLM HTTP endpoint serving | ❌ | 在 CANN 8.5 下所有 rollout 返回 503 |

## 5. Attention Benchmark 结果

在 Ascend 910 上测试了三种 attention 实现（bf16, HEAD_DIM=128, causal）：

| Seq Len | SDPA | npu_fusion (BNSD) | triton-ascend FA |
|---------|------|-------------------|------------------|
| 2k | **0.37ms** | 0.45ms | 7.24ms |
| 8k | **3.48ms** | 6.56ms | 91.11ms |
| 16k | **13.57ms** | 27.64ms | 364.56ms |
| 32k | **56.25ms** | 111.71ms | 1459.12ms |
| 64k | **224.07ms** | 446.16ms | 5847.34ms |

**结论：PyTorch SDPA 是最优选择**，`trainer.flash_attn=false` 即可。
triton-ascend FA 在 HEAD_DIM=128 下因 UB 容量限制被迫用小 block size，性能很差。

## 6. 环境搭建步骤（CANN 8.3 机器）

### Step 1: 创建 conda 环境
```bash
conda create -n codescout-npu python=3.12 -y
```

### Step 2: 安装 torch + torch_npu
```bash
pip install torch-npu==2.8.0.post2 numpy pyyaml
```

### Step 3: 安装 vllm + vllm-ascend
```bash
pip install vllm==0.11.0 --no-deps
# vllm-ascend 需要从 CANN 8.3 环境复制或从源码构建
pip install vllm-ascend==0.11.0rc1
```

### Step 4: 安装 vllm 运行时依赖
```bash
pip install msgspec pyzmq blake3 cbor2 cloudpickle compressed-tensors \
  depyf diskcache einops lark partial-json-parser prometheus-client \
  prometheus-fastapi-instrumentator psutil py-cpuinfo pybase64 setproctitle \
  numba scipy sentencepiece tiktoken openai-harmony gguf mistral_common
```

### Step 5: 安装 SkyRL 训练依赖
```bash
pip install hydra-core==1.3.2 omegaconf accelerate peft wandb datasets \
  tensordict jaxtyping polars torchdata func_timeout loguru tqdm ninja \
  tensorboard "transformers>=4.57,<5" "ray==2.51.1" hf_transfer gcsfs \
  litellm==1.82.6 docker tenacity authlib s3fs pybind11 "setuptools<76"
```

### Step 6: 安装 SkyRL（旧版）
```bash
git clone https://github.com/yucc-leon/SkyRL.git
cd SkyRL && git checkout codescout-npu  # 或 81e5a97
pip install -e skyrl-train/ --no-deps --no-build-isolation
pip install -e skyrl-gym/ --no-deps --no-build-isolation
```

### Step 7: 安装 openhands SDK
```bash
pip install openhands-tools openhands-agent-server openhands-workspace \
  openhands-sdk --no-deps --ignore-requires-python
pip install agent-client-protocol deprecation "fakeredis[lua]" fastmcp lmnr \
  python-frontmatter sqlalchemy wsproto fastapi alembic python-json-logger \
  binaryornot aiosqlite libtmux bashlex tom-swe
```

### Step 8: 安装 NPU 支持
```bash
# 应用源码 patch
python /path/to/npu_device_patch.py SkyRL/skyrl-train/skyrl_train

# 安装 npu_support 到 site-packages
cp -r SkyRL/npu_support $(python -c "import site; print(site.getsitepackages()[0])")/npu_support
cp SkyRL/npu_support/npu_autoload.pth $(python -c "import site; print(site.getsitepackages()[0])")/

# 修复 codescout agent
# 在 codescout/src/agent/agent.py 的 _initialize 方法末尾加:
#   self._initialized = True
```

### Step 9: 启动训练
```bash
cd codescout
bash scripts/run_async_training_npu.sh \
  -m /path/to/Qwen3-4B-Instruct-2507 \
  -n 4 -b 4 -c 1 -i 4 -t 4 \
  -d ./data/swe_smith/ \
  -s /path/to/ckpts \
  -r codescout-4b-npu
```

## 7. 关键文件清单

```
SkyRL/
├── npu_support/
│   ├── __init__.py
│   ├── patch_cuda.py          # 核心 monkey-patch
│   ├── npu_autoload.pth       # Python 启动时自动加载 patch
│   ├── setup_env.sh           # 一键环境搭建（需更新）
│   ├── bench_attn_v2.py       # Attention benchmark
│   └── README.md
├── skyrl_train/               # Import shim（仅新版 SkyRL 需要）
│   ├── __init__.py
│   ├── entrypoints/main_base.py
│   ├── generators/{base,utils,skyrl_gym_generator}.py
│   ├── inference_engines/{base,inference_engine_client,utils}.py
│   ├── utils/{__init__,trainer_utils,ppo_utils}.py
│   ├── training_batch.py
│   └── fully_async_trainer.py
└── skyrl-train/               # 旧版 SkyRL 训练代码（需 NPU patch）

codescout/
├── src/
│   ├── train_npu.py           # NPU 训练入口
│   └── agent/agent.py         # 需加 _initialized = True
├── scripts/
│   └── run_async_training_npu.sh  # NPU 训练脚本
├── configs/
│   └── reward_config_4b.yaml
└── docs/
    └── npu-adaptation-status.md   # 本文档
```

## 8. 已知问题和 TODO

1. **CANN 8.5 兼容性**: vllm-ascend 0.11 在 CANN 8.5 下 HTTP serving 不稳定（503）。
   需要 CANN 8.3 或升级到 vllm 0.16 + vllm-ascend 0.14（但需要解决 SkyRL config 不兼容）。

2. **wandb 连接**: Ray worker 里 wandb 需要代理。已在训练脚本和 SkyRL utils.py 里加了
   代理环境变量转发。如果仍超时，用 `trainer.logger=console` 或 `WANDB_MODE=offline`。

3. **Weight sync**: HCCL broadcast 在 CANN 8.3 上应该能工作（cann-recipes 验证过）。
   如果仍有问题，备选方案是实现 file-based weight sync。

4. **litellm 安全**: 锁定 litellm==1.82.6，不要升级到 1.82.7/1.82.8（供应链攻击）。

5. **NPU 僵尸进程**: 训练中断后 NPU 上的 vLLM/Ray 进程可能不会自动清理。
   训练前务必运行:
   ```bash
   ray stop --force
   pkill -9 -f "VLLMEngine|rayFSDP|EngineCore"
   npu-smi info  # 确认所有 NPU 无进程
   ```


## 9. Git 分支状态与切换指南

### 当前分支状态（2026-03-27）

**SkyRL 仓库** (`/sharedata/liyuchen/workspace/SkyRL`)
- remote: `https://github.com/yucc-leon/SkyRL` (origin)
- `main` 分支: 新版 SkyRL (commit 50ebff5)，支持 vllm 0.16，config 用 dataclass
- `codescout-npu` 分支: 旧版 (commit 81e5a97)，支持 vllm 0.11，config 用 YAML，codescout 兼容
- `stash@{0}`: 旧版上的 NPU 源码 patch（cuda→npu, nccl→hccl, GPU→NPU 等 7 个文件的改动）
- untracked: `npu_support/` 目录（monkey-patch）、`skyrl_train/` 目录（import shim，仅新版需要）

**codescout 仓库** (`/sharedata/liyuchen/workspace/codescout`)
- `main` 分支: 原始代码
- `npu-ascend-adapt` 分支（当前）: NPU 适配改动
  - `src/agent/agent.py`: 加了 `self._initialized = True`
  - `src/train_npu.py`: NPU 训练入口
  - `scripts/run_async_training_npu.sh`: NPU 训练脚本
  - `docs/npu-adaptation-status.md`: 本文档
  - `.env`: wandb API key

### 切换到 CANN 8.3 兼容状态

```bash
# 1. SkyRL: 切到旧版 + 恢复 NPU patch
cd /sharedata/liyuchen/workspace/SkyRL
git checkout codescout-npu       # 切到旧版 (81e5a97, vllm 0.11 兼容)
git stash pop                    # 恢复 NPU 源码 patch (cuda→npu 等)

# 2. codescout: 不用动，npu-ascend-adapt 分支兼容旧版 SkyRL

# 3. 验证
git log --oneline -1             # 应该显示: 81e5a97 fix the reasoning parser...
git diff --stat HEAD             # 应该显示 7 个文件的 NPU patch
```

### 如果在新机器上从零开始

```bash
# 克隆仓库
git clone https://github.com/yucc-leon/SkyRL.git
cd SkyRL

# 获取旧版 commit（从 codescout 作者的 fork）
git remote add upstream-codescout https://github.com/adityasoni9998/SkyRL.git
git fetch upstream-codescout 81e5a97c7430503c0c4e6508497cc5aa01a0c624 --depth=1
git checkout -b codescout-npu 81e5a97c7430503c0c4e6508497cc5aa01a0c624

# 应用 NPU 源码 patch（用自动化脚本）
python npu_support/apply_npu_patches.py skyrl-train/skyrl_train

# 安装
pip install -e skyrl-train/ --no-deps --no-build-isolation
pip install -e skyrl-gym/ --no-deps --no-build-isolation
```

### conda 环境清单

当前机器上创建了以下环境：

| 环境名 | Python | 用途 | 状态 |
|--------|--------|------|------|
| `codescout` | 3.11 | 旧的 vllm 0.11 环境（从系统预装） | 可用但缺训练依赖 |
| `codescout-npu` | 3.12 | CANN 8.5 适配尝试 | torch 2.9, vllm 已卸载 |
| `codescout-npu-v2` | 3.11 | v013 栈尝试 (vllm 0.16) | openhands patched, SkyRL config 不兼容 |
| `vllm-ascend-cann8.5-v013` | 3.11 | 已验证能 serve 的推理环境 | vllm 0.16 + vllm-ascend 0.14 |

**CANN 8.3 机器上建议只创建一个环境：`codescout-npu`，Python 3.12。**

## 10. 自动化 NPU Patch 脚本

将以下内容保存为 `SkyRL/npu_support/apply_npu_patches.py`，
用于在旧版 SkyRL 源码上自动应用 NPU 适配：

```python
"""Apply NPU device patches to SkyRL source files.
Usage: python apply_npu_patches.py <skyrl_train_dir>
Example: python apply_npu_patches.py SkyRL/skyrl-train/skyrl_train
"""
import sys, os

def get_device_name():
    try:
        import torch
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return 'npu'
    except: pass
    return 'cuda'

DEVICE = get_device_name()
if DEVICE == 'cuda':
    print("CUDA detected, no patching needed"); sys.exit(0)

base = sys.argv[1]
patches = {
    "distributed/fsdp_utils.py": [
        ('x = x.to_empty(device=torch.cuda.current_device(), recurse=False)',
         'x = x.to_empty(device=torch.npu.current_device(), recurse=False)'),
        ('torch.cuda.empty_cache()', 'torch.npu.empty_cache()'),
        ('device_id = torch.cuda.current_device()\n',
         'device_id = torch.npu.current_device()\n'),
        ('handle.flat_param_to(torch.device(f"cuda:{device_id}"), non_blocking=True)',
         'handle.flat_param_to(torch.device(f"npu:{device_id}"), non_blocking=True)'),
        ('device = torch.cuda.current_device()\n    model.to(device',
         'device = torch.npu.current_device()\n    model.to(device'),
        ('full_param = full_param.detach().cuda()',
         'full_param = full_param.detach().npu()'),
        ('full_tensor = torch.empty(sharded_param.size(), device="cuda"',
         'full_tensor = torch.empty(sharded_param.size(), device="npu"'),
        ('device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,)',
         'device_mesh = init_device_mesh("npu", mesh_shape=(world_size,)'),
        ('device_mesh = init_device_mesh(\n            "cuda"',
         'device_mesh = init_device_mesh(\n            "npu"'),
    ],
    "distributed/fsdp_strategy.py": [
        ('torch.cuda.manual_seed_all(seed)', 'torch.npu.manual_seed_all(seed)'),
        ('torch.cuda.set_device(local_rank)', 'torch.npu.set_device(local_rank)'),
        ('torch.cuda.synchronize()', 'torch.npu.synchronize()'),
        ('torch.cuda.empty_cache()', 'torch.npu.empty_cache()'),
        ('device_id=torch.cuda.current_device()',
         'device_id=torch.npu.current_device()'),
        ('load_fsdp_optimizer(optimizer, torch.cuda.current_device())',
         'load_fsdp_optimizer(optimizer, torch.npu.current_device())'),
    ],
    "distributed/strategy.py": [
        ('data = data.to(torch.cuda.current_device())',
         'data = data.to(torch.npu.current_device())'),
        ('ret = [torch.zeros_like(data).to(torch.cuda.current_device())',
         'ret = [torch.zeros_like(data).to(torch.npu.current_device())'),
        ('dist.all_gather(ret, data.to(torch.cuda.current_device()))',
         'dist.all_gather(ret, data.to(torch.npu.current_device()))'),
        ('if torch.cuda.is_available() and torch.cuda.device_count() > 0:',
         'if torch.npu.is_available() and torch.npu.device_count() > 0:'),
        ('rng_state["cuda"] = torch.cuda.get_rng_state()',
         'rng_state["npu"] = torch.npu.get_rng_state()'),
        ('"cuda" in rng_state and torch.cuda.is_available()',
         '"npu" in rng_state and torch.npu.is_available()'),
        ('torch.cuda.set_rng_state(rng_state["cuda"])',
         'torch.npu.set_rng_state(rng_state["npu"])'),
    ],
    "workers/fsdp/fsdp_worker.py": [
        ('torch.distributed.get_rank() % torch.cuda.device_count()',
         'torch.distributed.get_rank() % torch.npu.device_count()'),
        ('torch.cuda.empty_cache()', 'torch.npu.empty_cache()'),
        ('device = torch.cuda.current_device()',
         'device = torch.npu.current_device()'),
    ],
    "workers/worker.py": [
        ('backend="nccl"', 'backend="hccl"'),
        ('{"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node}',
         '{"NPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node}'),
    ],
    "inference_engines/vllm/vllm_engine.py": [
        ('backend="nccl"', 'backend="hccl"'),
        ('device="cuda"', 'device="npu"'),
        ('torch.cuda.current_device()', 'torch.npu.current_device()'),
        ('torch.cuda.get_device_properties', 'torch.npu.get_device_properties'),
    ],
    "model_wrapper.py": [
        ('from flash_attn.bert_padding import pad_input, unpad_input',
         'try:\n    from flash_attn.bert_padding import pad_input, unpad_input\n'
         'except ImportError:\n    pad_input = None\n    unpad_input = None'),
    ],
}

for relpath, replacements in patches.items():
    fpath = os.path.join(base, relpath)
    if not os.path.exists(fpath):
        print(f"  SKIP {relpath}"); continue
    with open(fpath, 'r') as f: content = f.read()
    changed = sum(1 for old, new in replacements if old in content)
    for old, new in replacements: content = content.replace(old, new)
    with open(fpath, 'w') as f: f.write(content)
    print(f"  {relpath}: {changed}/{len(replacements)}")
print("Done!")
```
