# 昇腾 NPU 上长序列 RL 训练的性能悬崖：根因分析与解决方案

## 摘要

在昇腾 910 (64 GB HBM) 上使用 SkyRL + FSDP2 对 Qwen3-4B 进行 RL 训练时，
我们观察到 `policy_train` 耗时随序列长度出现断崖式跳变：glen < 34000 时
per-iteration 约 12-34s，glen > 34000 时突然跳到 56-77s，最慢的 step 达到
1244s（正常 step 的 8.7 倍）。

经过系统排查，我们定位到根因是 `lm_head` 产生的 logits tensor `(B, S, V)` 及其
后续的 `log_softmax` 计算在长序列时占用大量显存，叠加 FSDP 参数 all-gather 和
gradient checkpointing activations，总显存逼近 64 GB 上限，导致 FSDP 的
compute-communication overlap 退化。

最终通过启用 Ulysses Sequence Parallel (SP=2) 解决，训练耗时波动从 8.7x 降至 2.7x。

## 1. 问题现象

### 1.1 训练配置

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3-4B-Instruct (H=2560, V=151936, L=36) |
| 硬件 | 昇腾 910, 64 GB HBM, 8 卡 (4 训练 + 4 推理) |
| 分布式策略 | FSDP2, DP=4, gradient_checkpointing=True, cpu_offload=True |
| 序列长度 | max_prompt_length=40960, max_generate_length=8192 |
| RL 算法 | GSPO (eps_clip_low=0.0003, eps_clip_high=0.0004) |
| 训练框架 | SkyRL (基于 verl) |

### 1.2 现象描述

从训练 log (`0401_1643_Qwen3-4B-Instruct-2507-npu-8x8.log`) 提取 135 个 training
step 的数据：

- `policy_train` 耗时范围：143s ~ 1244s
- max/min 比值：8.7x
- 标准差：366s

耗时与 `glen`（batch 内 padding 后的最大序列长度）强相关：

| glen 区间 | 步数 | 平均 per-iteration | 平均 ms/token |
|-----------|------|-------------------|---------------|
| 10000-15000 | 20 | 12.2s | 12.7 |
| 15000-20000 | 37 | 15.7s | 13.0 |
| 20000-25000 | 46 | 21.4s | 14.1 |
| 25000-30000 | 48 | 27.4s | 15.3 |
| 30000-35000 | 40 | 40.0s | 16.9 |
| **35000-40000** | **56** | **76.5s** | **31.9** |

glen 30000-35000 到 35000-40000 时，ms/token 从 16.9 跳到 31.9，接近 2x。

### 1.3 对比：H200 上无此问题

同样的训练配置在 H200 (80 GB) 上 `policy_train` 稳定在 ~250s，无 spike。

## 2. 排查过程

### 2.1 假设与排除

| 假设 | 验证方法 | 结果 |
|------|----------|------|
| SDPA 在 NPU 上有性能悬崖 | 独立 SDPA benchmark (8K-41K) | ❌ 增长完全平滑，无悬崖 |
| 单层/多层 transformer 计算有悬崖 | 单层 + 36 层 benchmark | ❌ 平滑，无悬崖 |
| 显存压力导致计算变慢 | 占用 48/64 GB 后测计算性能 | ❌ 不影响 |
| HCCL 通信有性能拐点 | all-reduce/all-gather/reduce-scatter 带宽测试 | ❌ 全范围稳定 ~175 GB/s |
| FSDP overhead 在长序列时跳变 | 对比训练 per-it 和单卡 benchmark | ✅ glen < 34K 时 1.4x，glen > 34K 时 3x |

### 2.2 独立 SDPA Benchmark

在 CANN 8.3 RC1 上测试 SDPA forward+backward (B=1, H=32, D=128, bf16)：

| S | SDPA fwd+bwd (ms) | NFA fwd+bwd (ms) | SDPA/NFA |
|---|-------------------|-------------------|----------|
| 8192 | 12.4 | 22.0 | 0.56x |
| 16384 | 45.4 | 87.3 | 0.52x |
| 32768 | 176.8 | 344.2 | 0.51x |
| 34816 | 199.0 | 389.0 | 0.51x |
| 36864 | 222.2 | 433.5 | 0.51x |
| 40960 | 273.5 | 538.5 | 0.51x |

结论：SDPA 增长完全平滑（相邻点 normalized growth ≈ 1.03x），且比 `npu_fusion_attention` 快 2x。
SDPA 只占训练 per-iteration 时间的 10-18%，不是瓶颈。

### 2.3 HCCL 带宽测试 (4 NPU)

| 操作 | 带宽 (GB/s) | 稳定性 |
|------|------------|--------|
| all-reduce | ~188 | 100MB+ 稳定 |
| all-gather | ~175 | 全范围稳定 |
| reduce-scatter | ~150 | 全范围稳定 |

通信本身无拐点。

### 2.4 FSDP Overhead 分析

对比训练 per-iteration 时间和单卡 36 层 benchmark：

| glen | 单卡 36L (ms) | 训练 per-it (ms) | ratio | overhead |
|------|-------------|----------------|-------|----------|
| 15882 | 4867 | 12500 | 2.57x | 1.57x |
| 26000 | 10102 | 23850 | 2.36x | 1.36x |
| 32770 | 14412 | 34400 | 2.39x | 1.39x |
| **34000** | **15313** | **56000** | **3.66x** | **2.66x** |
| 36000 | 16748 | 71000 | 4.24x | 3.24x |
| 37544 | 17850 | 73800 | 4.13x | 3.13x |

glen < 34000 时 FSDP overhead 稳定在 ~1.4x，glen > 34000 时跳到 ~3x。
单卡计算无悬崖，通信无拐点，说明问题在 FSDP 的 compute-communication overlap。

## 3. 根因定位

### 3.1 显存分解

通过逐项实测（单卡，S=34000）：

| 组件 | 大小 | 说明 |
|------|------|------|
| logits tensor `(1, S, V)` bf16 | 10.3 GB | `lm_head(hidden_states)` 的输出 |
| `log_softmax` fp32 中间结果 | 20.7 GB | GSPO 要求 fp32 精度 |
| backward autograd 保存 | ~10 GB | logits 在 backward 期间不能释放 |
| grad ckpt activations | ~15 GB | 36 层的 checkpoint hidden states |
| FSDP 参数 all-gather | ~0.2 GB | 单层临时全量 |
| FSDP 梯度 (sharded) | 2 GB | |
| 其他 buffer | ~2 GB | |

**logits + log_softmax + backward 合计 ~41 GB**，占单卡 64 GB 的 64%。

### 3.2 为什么 GSPO 更严重

GSPO 的 clip range 极窄 `[0.9997, 1.0004]`，对应 log ratio `[-0.0003, 0.0004]`。
bf16 的 `log_softmax` 相比 fp32 有 ~0.015 的平均误差，比 clip range 大 50 倍。
如果不 upcast fp32，99% 的 token 会被数值噪声 clip 掉，梯度信号完全消失。

因此 SkyRL 的 `logprobs_from_logits_v2` 在 bf16 分支强制 upcast 到 fp32：

```python
# SkyRL/skyrl-train/skyrl_train/utils/torch_utils.py
row_logprobs = F.log_softmax(row_logits.float(), dim=-1)  # bf16 → fp32
```

这使得 `log_softmax` 的中间结果从 10.3 GB (bf16) 膨胀到 20.7 GB (fp32)。

标准 PPO/GRPO 的 clip range 是 `[0.8, 1.2]`，bf16 精度足够，不需要 fp32 upcast。
但即使不 upcast，S=40000 时 bf16 logits + log_softmax 仍有 ~20 GB，在 64 GB NPU 上
依然紧张。

### 3.3 为什么 H200 没问题

H200 有 80 GB 显存，比 NPU 多 16 GB。同样的 41 GB logits 开销在 80 GB 上余量 39 GB，
在 64 GB 上余量只有 23 GB。叠加 FSDP 参数和 activations 后，NPU 逼近上限而 H200 仍有余量。

### 3.4 根因总结

**logits tensor `(B, S, V)` 是 RL 训练中的隐藏显存炸弹。** 它不被 FSDP 分片，
大小随序列长度线性增长。当 logits + 中间结果逼近单卡显存上限时，FSDP 的
compute-communication overlap 退化（无法提前 prefetch 下一层参数），
通信从并行变成串行，per-iteration 时间翻倍。

GSPO 的 fp32 upcast 是加重因素（显存翻倍），但不是根因。根因是 `S × V` 的
logits tensor 本身。

## 4. 解决方案

### 4.1 方案选择

| 方案 | 改动 | 效果 | 状态 |
|------|------|------|------|
| 限制 max_response_len < 34K | 改配置 | 直接避开悬崖 | 不可接受（截断轨迹） |
| Chunked logprobs | 改训练框架 | 从根本上消除 logits 显存 | FSDP2 DTensor 不兼容 |
| Sequence Parallel (SP=2) | 改配置 + 小 patch | 序列维度减半 | ✅ 已验证 |

### 4.2 Chunked Logprobs 的尝试与失败

思路：不一次性 materialize 完整 `(B, S, V)` logits，而是分 chunk 计算 `lm_head` +
`log_softmax` + `gather`，每次只 materialize `(B, 1024, V)`。

单卡验证结果：
- 显存：57.1 GB → 2.55 GB（节省 54.6 GB）
- 计算开销：+2-5%（可忽略）

但在 FSDP2 训练中连续失败 4 次，全部因为 DTensor 兼容性问题：
- `nn.Identity()` 替换 lm_head → DTensor 追踪断裂
- `register_forward_pre_hook` 截获 hidden_states → DTensor slice 不兼容
- `output_hidden_states=True` + `logits_to_keep=1` → hidden_states 是 DTensor，后续 lm_head 调用报错
- 直接用 `lm_head.weight` 做 `F.linear` → FSDP reshard 后权重不可用

核心限制：FSDP2 用 DTensor 包装所有参数，任何在 forward 上下文之外调用 wrapped module
的操作都会报 `mixed torch.Tensor and DTensor` 错误。

### 4.3 Sequence Parallel 方案

#### 原理

Ulysses SP 把序列维度分片到多卡。SP=2 时每卡只处理 S/2 长度的序列，
logits tensor 从 `(1, S, V)` 变成 `(1, S/2, V)`，显存减半。

SP 通过 all-to-all 通信在 attention 前后交换数据：
- forward: scatter heads, gather sequence → 每卡处理全部 heads 但只有 S/SP 的序列
- attention 后: gather heads, scatter sequence → 恢复原始 layout

#### 实现

SkyRL 已内置 Ulysses SP 支持，但在 NPU 上被两个问题阻塞：

1. `flash_attn.bert_padding.unpad_input/pad_input` 被 stub 为 `raise NotImplementedError`
2. SP 依赖 `use_sample_packing=True`，后者依赖 `flash_attention_2`

解决：在 `SkyRL/npu_support/patch_cuda.py` 中实现 `unpad_input`/`pad_input` 的纯 PyTorch 版本：

```python
def _unpad_input(hidden_states, attention_mask):
    import torch as _t
    seqlens = attention_mask.sum(dim=-1, dtype=_t.int32)
    max_seqlen = seqlens.max().item()
    indices = _t.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = _t.zeros(seqlens.shape[0] + 1, dtype=_t.int32, device=seqlens.device)
    cu_seqlens[1:] = seqlens.cumsum(0)
    batch, seqlen = hidden_states.shape[:2]
    other_dims = hidden_states.shape[2:]
    hidden_states_flat = hidden_states.reshape(batch * seqlen, *other_dims)
    hidden_states_unpad = hidden_states_flat.index_select(0, indices)
    return hidden_states_unpad, indices, cu_seqlens, max_seqlen, cu_seqlens

def _pad_input(hidden_states, indices, batch, seqlen):
    import torch as _t
    other_dims = hidden_states.shape[1:]
    output = _t.zeros(batch * seqlen, *other_dims, dtype=hidden_states.dtype, device=hidden_states.device)
    output.index_copy_(0, indices, hidden_states)
    return output.view(batch, seqlen, *other_dims)
```

训练配置变更：

```bash
trainer.flash_attn=true              # 启用 flash_attention_2 (NPU 上底层仍走 SDPA/npu_fusion_attention)
trainer.use_sample_packing=true      # SP 的前置依赖
trainer.policy.sequence_parallel_size=2  # SP=2, DP 从 4 降到 2
```

注：`flash_attn=true` 在 NPU 上不改变 attention kernel（底层都是 `npu_fusion_attention`），
只是满足 SkyRL 代码中 SP → sample_packing → flash_attention_2 的依赖链。

#### 验证结果

67 个 training step 的对比：

| 指标 | 旧 (无 SP, DP=4) | 新 (SP=2, DP=2) |
|------|-------------------|-------------------|
| min | 143s | 83s |
| max | 1244s | 228s |
| max/min | 8.7x | 2.7x |
| stdev | 366s | 37s |
| mean | 536s | 132s |

最慢 step 从 1244s 降到 228s，波动从 8.7x 降到 2.7x。
剩余的 2.7x 波动是正常的序列长度差异（avg_response_length 从 2565 到 12578）。

#### 代价

- DP 从 4 降到 2：梯度估计方差可能增大，但 sample packing 去掉了 padding 浪费，实际有效 token 数可能更多
- SP 的 all-to-all 通信：每层 ~0.5ms，36 层共 ~36ms，占 per-iteration 的 1-2%
- sample packing 改变 attention 计算方式：理论等价，但 loss 计算有微小差异

## 5. 通用判断方法

### 5.1 经验法则

对于昇腾 910 (64 GB)，当 `S_max × V × 2 bytes` 超过单卡显存的 15% 时考虑 SP：

```
阈值 = 64 GB × 15% = 9.6 GB
S_max > 9.6e9 / V / 2
```

对于 Qwen3 系列 (V ≈ 152K)：`S_max > 31500` 时需要 SP。

### 5.2 Dry-Run 工具

我们提供了 `rl_mem_dryrun_v2.py`，用单层 transformer + lm_head 的实测数据估算
FSDP 训练的 peak 显存，不需要跑完整训练：

```bash
python rl_mem_dryrun_v2.py \
  --model /path/to/model \
  --max_seq_len 8192,16384,32768,40960 \
  --train_gpus 4
```

输出示例（Qwen3-4B, 4 卡）：

```
     S   fwd_peak   bwd_peak    + FSDP      余量  状态
  8192      22.3G      27.2G     25.1G   40.6G  ✓ 安全
 16384      36.4G      46.3G     44.3G   21.5G  ✓ 安全
 24576        OOM        OOM       OOM      <0  ✗ OOM

⚠ 悬崖点: S ≈ 16384 ~ 24576
推荐: → SP=2 (DP=2)
```

注意：dry-run 的悬崖点比实际训练偏保守约 15%（因为单卡放了完整模型参数，
而 FSDP 训练时参数是分片的），但对"是否需要 SP"的判断是准确的。

### 5.3 不同模型 size 的策略参考

基于实测数据的估算（昇腾 910, 64 GB, cpu_offload=True, grad_ckpt=True）：

| 模型 | 8K | 16K | 32K | 40K |
|------|-----|------|------|------|
| 1.7B | FSDP | FSDP | SP2-4 | SP4+ |
| 4B | FSDP | FSDP | SP2-4 | SP4+ |
| 8B | FSDP | FSDP | SP4+ | SP4+ |
| 14B | FSDP | SP2 | SP4+ | SP8/TP |
| 32B | SP2 | SP4+ | SP8/TP | SP8/TP |
| 72B | SP4+ | SP8/TP | TP+SP | TP+SP |

注：
- "FSDP" = 纯 FSDP 即可，不需要 SP
- "SP2-4" = 需要 SP，具体度数取决于卡数和 offload 配置
- "SP8/TP" = SP 不够，需要 Tensor Parallel 或更多卡
- GSPO 的 fp32 upcast 会让悬崖提前 ~15%，标准 PPO/GRPO 可以推后

## 6. 未来优化方向

### 6.1 Fused Linear Logprobs

从根本上消除 logits 显存的方案：在 FSDP forward 上下文内，用 fused kernel 直接从
hidden_states 计算 per-token logprobs，不 materialize 完整 `(B, S, V)` logits。

相关工作：
- [Cut Cross-Entropy (CCE)](https://arxiv.org/abs/2411.09009)：SFT 场景下将 loss 计算显存从 24 GB 降到 1 MB
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)：`FusedLinearCrossEntropy` 实现了 CCE 思路
- [Selective Log-Softmax](https://tylerromero.com/posts/2025-02-selective-log-softmax/)：verl/SkyRL 已采用的 per-token 优化

当前空白：没有框架实现了 RL 场景的 fused_linear_logprobs（输出 per-token logprobs
而非 scalar loss），且需要兼容 FSDP2 + DTensor。

### 6.2 triton-ascend 实现

用 triton-ascend 在昇腾上写 fused_linear_logprobs kernel，在 SRAM 中完成
`lm_head matmul + log_softmax + gather`，全程不将完整 logits 写入 HBM。
这是最根本的解决方案，但开发成本较高。

## 附录

### A. 分析脚本

| 脚本 | 用途 |
|------|------|
| `analyze_spike.py` | 从训练 log 提取 per-step 数据，关联 glen 和耗时 |
| `analyze_cliff.py` | 精确定位性能悬崖 |
| `analyze_cliff_v2.py` | 分离 SDPA vs non-SDPA 耗时 |
| `bench_sdpa_cliff.py` | 独立 SDPA/NFA benchmark |
| `bench_layer_cliff.py` | 单层 transformer benchmark |
| `bench_multilayer_cliff.py` | 36 层 + gradient checkpointing benchmark |
| `bench_mem_pressure.py` | 显存压力对计算性能的影响 |
| `bench_hccl_bw.py` | HCCL 集合通信带宽测试 |
| `bench_chunked_overhead.py` | Chunked logprobs 的计算开销测试 |
| `rl_mem_dryrun_v2.py` | 显存 dry-run 工具 |
| `rl_mem_survey.py` | 批量模型 size 显存估算 |

### B. 训练数据来源

- 训练 log: `/sharedata/liyuchen/workspace/logs/0401_1643_Qwen3-4B-Instruct-2507-npu-8x8.log`
- SP=2 验证 log: `codescout/logs/sp2_v3.log`
- 135 个 step (无 SP) + 67 个 step (SP=2) 的完整数据

### C. 代码变更

1. `SkyRL/npu_support/patch_cuda.py`：实现 `unpad_input`/`pad_input` 纯 PyTorch 版本
2. `codescout/scripts/run_async_training_npu_sp2.sh`：SP=2 训练脚本
