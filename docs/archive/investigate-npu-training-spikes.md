# 排查 NPU 训练耗时 Spike 问题

## ✅ 已解决

**问题：** `policy_train` 在 NPU 上波动 143s ~ 1244s（8.7x），glen > 34000 时耗时翻倍。

**根因：** `lm_head` 产生的 logits tensor `(B, S, V)` 在长序列时占 ~11 GB，叠加 FSDP 参数
all-gather 和 gradient checkpointing activations，总显存接近 64 GB 上限，导致 FSDP 的
compute-communication overlap 退化。

**解决方案：** 启用 Ulysses Sequence Parallel (SP=2)，每卡只处理 S/2 长度的序列，
logits 显存从 ~11 GB 降到 ~5.5 GB。

**改动：**
1. `SkyRL/npu_support/patch_cuda.py`（及 site-packages 副本）：实现 `unpad_input` / `pad_input`
   的纯 PyTorch 版本，替换原来的 `raise NotImplementedError` stub
2. 训练脚本：`flash_attn=true`, `use_sample_packing=true`, `sequence_parallel_size=2`

**效果（67 step 验证）：**

| 指标 | 旧（无 SP, DP=4） | 新（SP=2, DP=2） |
|------|-------------------|-------------------|
| min | 143s | 83s |
| max | 1244s | 228s |
| max/min | 8.7x | 2.7x |
| stdev | 366s | 37s |

---

## 原始现象

`timing/policy_train` 在 NPU 上波动剧烈（250s ~ 1200s），而 H200 上稳定在 ~250s。
Spike 不是恒定的倍数关系，说明不是纯算力差异，而是某些 batch 触发了性能退化。

## 假设

| # | 假设 | 验证方法 |
|---|------|----------|
| 1 | 长序列导致 SDPA 性能退化 | 关联 policy_train 耗时 vs batch 序列长度 |
| 2 | 显存分配器频繁扩展/收缩 | 监控 NPU 显存分配事件 |
| 3 | HCCL gradient sync 偶发卡顿 | 分离 compute 和 communication 耗时 |
| 4 | SDPA 在 NPU 上对特定序列长度有性能悬崖 | 独立 benchmark 不同长度的 SDPA |

## 排查步骤

### Step 1: 从 wandb/log 提取 per-step 数据

从 Ray worker log 提取每个 training step 的耗时和序列长度：

```bash
# 找到 skyrl_entrypoint 的 worker log
F=/tmp/ray/ray/session_*/logs/worker-*<skyrl_entrypoint_pid>*.err

# 提取 policy_train 耗时
grep "policy_train.*time cost" $F | \
  sed 's/.*time cost: \([0-9.]*\)s/\1/' > /tmp/policy_train_times.txt

# 提取 avg_response_length（proxy for sequence length）
grep "avg_response_length" $F | \
  sed 's/.*avg_response_length: \([0-9.]*\)/\1/' > /tmp/response_lengths.txt

# 简单关联分析
paste /tmp/policy_train_times.txt /tmp/response_lengths.txt | \
  awk '{print $1, $2}' | sort -k1 -rn | head -20
# 看 policy_train 最慢的 20 个 step 是否都对应长序列
```

如果有 wandb 数据，直接在 wandb UI 里把 `timing/policy_train` 和
`generate/max_num_tokens` 叠在一起看相关性。

### Step 2: SDPA 序列长度 benchmark

独立测试 SDPA 在不同序列长度下的性能，看是否有性能悬崖：

```python
"""SDPA 序列长度 vs 耗时 benchmark"""
import torch, torch_npu, time

torch.npu.set_device(0)  # 用空闲的 NPU
D, H, B = 128, 32, 1  # Qwen3-4B config
DTYPE = torch.bfloat16

results = []
for S in [1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, 40960]:
    q = torch.randn(B, H, S, D, dtype=DTYPE, device="npu")
    k = torch.randn(B, H, S, D, dtype=DTYPE, device="npu")
    v = torch.randn(B, H, S, D, dtype=DTYPE, device="npu")

    # Warmup
    for _ in range(3):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.npu.synchronize()

    # Forward
    t0 = time.perf_counter()
    for _ in range(5):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.npu.synchronize()
    fwd_ms = (time.perf_counter() - t0) / 5 * 1000

    # Forward + Backward
    q.requires_grad_(True)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        loss = out.sum()
        loss.backward()
        q.grad = None
    torch.npu.synchronize()
    fwd_bwd_ms = (time.perf_counter() - t0) / 5 * 1000

    print(f"S={S:>6}  fwd={fwd_ms:>8.1f}ms  fwd+bwd={fwd_bwd_ms:>8.1f}ms  ratio={fwd_bwd_ms/fwd_ms:.2f}x")
    results.append((S, fwd_ms, fwd_bwd_ms))
    del q, k, v
    torch.npu.empty_cache()

# 检查是否有非线性增长（性能悬崖）
print("\n=== 增长率 ===")
for i in range(1, len(results)):
    s_ratio = results[i][0] / results[i-1][0]
    t_ratio = results[i][2] / results[i-1][2]
    print(f"S {results[i-1][0]}->{results[i][0]} ({s_ratio:.1f}x): time {t_ratio:.2f}x")
```

预期：如果 SDPA 是 O(n) 显存 + O(n²) 计算，耗时应该随 S² 增长。
如果某个长度区间增长率突然变大，说明有性能悬崖。

### Step 3: 显存分配监控

在训练脚本里加环境变量，让 PyTorch 记录显存分配：

```bash
# 方法 1: 显存快照（会影响性能，只用于诊断）
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,record_history:True

# 在训练代码里（或 monkey-patch）加：
# torch.npu.memory._record_memory_history(max_entries=100000)
# ... 跑几个 step ...
# torch.npu.memory._dump_snapshot("/tmp/npu_mem_snapshot.pickle")
# 然后用 https://pytorch.org/memory_viz 可视化
```

```bash
# 方法 2: 简单监控（不影响性能）
# 在训练循环里每个 step 前后打印：
# torch.npu.memory_allocated() / 1e9
# torch.npu.memory_reserved() / 1e9
# torch.npu.max_memory_allocated() / 1e9
```

看 spike step 的 `max_memory_allocated` 是否显著高于正常 step。

### Step 4: 分离 compute 和 communication

在 SkyRL 的 `fsdp_strategy.py` 里加计时：

```python
# backward 前后
torch.npu.synchronize()
t0 = time.perf_counter()
loss.backward()
torch.npu.synchronize()
backward_time = time.perf_counter() - t0

# optimizer step 前后
torch.npu.synchronize()
t0 = time.perf_counter()
optimizer.step()
torch.npu.synchronize()
optimizer_time = time.perf_counter() - t0

logger.info(f"backward={backward_time:.2f}s, optimizer={optimizer_time:.2f}s")
```

如果 spike 主要在 backward（compute），说明是 SDPA/序列长度问题。
如果在 optimizer step（包含 FSDP all-reduce），说明是 HCCL 通信问题。

### Step 5: 对比 npu_fusion_attention

如果确认 SDPA 有性能悬崖，可以尝试用 `torch_npu.npu_fusion_attention` 替代。
需要在 HuggingFace 模型里注册自定义 attention：

```python
# 思路：monkey-patch Qwen3 的 attention forward
# 把 F.scaled_dot_product_attention 替换为 torch_npu.npu_fusion_attention
# 注意 layout 转换（BNSD vs BSH）的开销
```

这个改动比较大，只在确认 SDPA 是瓶颈后再做。

## 排查结果（2026-04-02）

### 数据来源

分析 log: `/sharedata/liyuchen/workspace/logs/0401_1643_Qwen3-4B-Instruct-2507-npu-8x8.log`
- 135 个 training step，8x8 配置（64 NPUs），DP=1
- policy_train 耗时范围: 143s ~ 1244s

### 结论：~~SDPA 在 glen ≈ 34000 处存在性能悬崖~~ → 非 SDPA 组件在 glen ≈ 34000 处存在性能悬崖

**假设 1 和假设 4 部分确认：spike 确实与序列长度强相关，但瓶颈不在 SDPA 本身。**

独立 SDPA benchmark（`bench_sdpa_cliff.py`）显示 SDPA 在 8192-40960 范围内增长平滑，无悬崖。
训练中 SDPA 只占 per-iteration 时间的 10-18%。

真正的悬崖在 non-SDPA 部分（MLP forward/backward、FSDP all-gather/reduce-scatter、
gradient accumulation 等），在 glen ≈ 34000 处 non-SDPA 耗时从 ~28s 跳到 ~58s（2x）。

排除的假设：
- 假设 3（HCCL 卡顿）：DP=1，不存在 DP 间的 straggler 问题。rank_0 worker 耗时与 entrypoint 总耗时 100% 匹配（overhead < 1.2s）。
- 假设 4（SDPA 性能悬崖）：独立 benchmark 确认 SDPA 无悬崖，增长平滑。

**新假设：FSDP 通信与计算的 overlap 在长序列时失效。**

### NPU Benchmark 结果（2026-04-03）

| 实验 | 结果 |
|------|------|
| 独立 SDPA benchmark (单层 attention) | 8192-40960 增长平滑，无悬崖。SDPA 比 NFA 快 2x |
| 单层 Transformer fwd+bwd | 8192-49152 增长平滑，无悬崖 |
| 36 层 + gradient checkpointing (单卡) | 16384-40960 增长平滑，无悬崖。peak 33 GB |
| 显存压力测试 (单卡) | 占用 48/65 GB 时计算性能不受影响 |
| FSDP2 4 卡 | S=16384 成功 (5980ms)，S>=24576 OOM (cross_entropy logits) |

所有单卡实验都没有复现悬崖。悬崖只在训练中出现，且：
- 训练 per-iteration / 单卡 benchmark 的比值在 glen < 34000 时稳定在 ~2.4x
- glen > 36000 时跳到 ~4.1x
- 额外的 1.7x 来自 FSDP 通信或 SkyRL 训练框架的某个组件

**根因已定位：lm_head logits tensor 在长序列时导致显存接近上限。**

### 根因分析（最终）

训练中 `HFModelWrapper.forward()` 会计算完整的 `logits_BSV = model(sequences)["logits"]`，
大小为 `(B, S, V)` 其中 V=151643（Qwen3 vocab size）。

| glen | logits 大小 (bf16) | 说明 |
|------|-------------------|------|
| 16384 | 4.7 GB | 正常 |
| 24576 | 7.0 GB | 正常 |
| 32768 | 9.4 GB | 接近临界 |
| 34816 | 10.0 GB | 临界区 |
| 37000 | 10.6 GB | 超过临界 → spike |
| 40960 | 11.7 GB | OOM 风险 |

训练中每卡显存布局（FSDP 4 卡，cpu_offload=true）：
- FSDP 参数分片: ~2 GB（backload 时临时 all-gather 到 ~8 GB）
- Optimizer states: ~4 GB（CPU offload 后可能更少）
- Gradient checkpointing activations: ~10-15 GB（取决于 glen）
- logits tensor: 4.7-11.7 GB（取决于 glen）
- logprobs_from_logits 中间结果: ~同 logits 大小
- chunked_entropy_from_logits 中间结果: ~同 logits 大小

当 glen > 34000 时，logits + 中间结果 ≈ 30 GB，加上其他占用超过 60 GB（64 GB 上限），
触发 NPU 显存分配器频繁 defragmentation 或 expandable segments 扩展，导致性能退化 2x。

验证证据：
- 独立 SDPA benchmark: 无悬崖（不涉及 logits）
- 单层/36 层 benchmark: 无悬崖（不涉及 lm_head）
- 显存压力测试: 纯计算不受影响，但分配器行为可能不同
- HCCL 带宽: 稳定，通信只占 119ms（理论值）
- HCCL 带宽实测 (4 NPU): all-reduce ~188 GB/s, all-gather ~175 GB/s, reduce-scatter ~150 GB/s，全范围稳定无拐点
- 训练 OOM: `Tried to allocate 23.19 GiB`（= logits + softmax 中间结果）
- FSDP overhead 分析: glen < 35000 时 ratio 2.5x，glen > 35000 时跳到 4.4x

**FSDP overhead 详细数据：**

| glen | 单卡 36L(ms) | 训练 per-it(ms) | ratio | overhead |
|------|-------------|----------------|-------|----------|
| 15882 | 4867 | 12500 | 2.57x | 1.57x |
| 26000 | 10102 | 23850 | 2.36x | 1.36x |
| 32770 | 14412 | 34400 | 2.39x | 1.39x |
| **34000** | **15313** | **56000** | **3.66x** | **2.66x ← 跳变** |
| 36000 | 16748 | 71000 | 4.24x | 3.24x |
| 37544 | 17850 | 73800 | 4.13x | 3.13x |

### 关键数据

**glen（batch 内 padding 后的 max 序列长度）与 policy_train 耗时强相关：**

| 指标 | 值 |
|------|-----|
| Pearson r(glen, rank0_time) | 0.8958 |
| Pearson r(avg_response_length, total_time) | 0.3383（弱，因为 avg 不反映 max） |

**性能悬崖位于 glen ≈ 34000，但不在 SDPA：**

独立 SDPA benchmark（CANN 8.3 RC1, B=1, H=32, D=128, bf16）：

| S | SDPA fwd+bwd (ms) | ms/token | 增长率 |
|---|-------------------|----------|--------|
| 8192 | 12.4 | 1.52 | - |
| 16384 | 45.4 | 2.77 | - |
| 24576 | 100.3 | 4.08 | - |
| 32768 | 176.8 | 5.39 | norm 1.03x |
| 33792 | 187.4 | 5.55 | norm 1.03x |
| 34816 | 199.0 | 5.71 | norm 1.03x |
| 35840 | 209.7 | 5.85 | norm 1.02x |
| 36864 | 222.2 | 6.03 | norm 1.03x |
| 37888 | 234.2 | 6.18 | norm 1.03x |
| 40960 | 273.5 | 6.68 | norm 1.08x |

→ SDPA 增长完全平滑，无悬崖。SDPA 只占训练 per-iteration 时间的 10-18%。

**训练中 non-SDPA 组件的悬崖：**

| glen 区间 | 步数 | 平均 per_it | SDPA 占比 | non-SDPA 耗时 |
|-----------|------|-------------|-----------|---------------|
| 10000-15000 | 20 | 12.2s | 9.3% | 11.1s |
| 15000-20000 | 37 | 15.7s | 12.0% | 13.8s |
| 20000-25000 | 46 | 21.4s | 14.5% | 18.3s |
| 25000-30000 | 48 | 27.4s | 16.6% | 22.9s |
| 30000-35000 | 40 | 40.0s | 16.4% | 33.8s |
| **35000-40000** | **56** | **76.5s** | **10.3%** | **68.6s ← 2x 跳变** |

glen 30000-35000 → 35000-40000 时，non-SDPA 从 33.8s 跳到 68.6s（2.03x），
而 SDPA 占比反而从 16.4% 降到 10.3%。

**注意：** `avg_response_length` 不能反映 spike，因为 padding 到 batch 内最长序列。
真正决定计算量的是 `glen`（tqdm 输出中的 `glen=` 字段），即 padding 后的 max response length。

### 根因分析

~~SkyRL 的 `convert_prompts_responses_to_batch_tensors()` 将所有序列 padding 到 batch 内的
`max_input_len + max_output_len`。当 batch 内有一个特别长的序列时，所有 64 个序列都被
padding 到该长度。SDPA 在 NPU 上对 seq_len > 34000 存在性能悬崖（ms/token 翻倍），
导致 policy_train 耗时从 ~500s 跳到 ~1200s。~~

**更新（NPU benchmark 后）：** 独立 SDPA benchmark 确认 SDPA 本身无性能悬崖。
悬崖来自 non-SDPA 组件（MLP forward/backward、FSDP all-gather/reduce-scatter 等）。

可能的根因：
1. **显存压力**：glen > 34000 时 activation memory 超过 NPU 显存阈值，触发显存分配器
   频繁扩展/收缩或 swap。FSDP2 的 activation 不分片，全部存在本地。
2. **MLP 计算**：Qwen3-4B 的 MLP intermediate_size=18944，glen=37000 时单层 MLP 的
   activation 约 37000 × 18944 × 2 bytes ≈ 1.3 GB，36 层共 ~47 GB（gradient checkpointing
   下只保留部分层）。
3. **FSDP all-gather 与计算重叠失效**：长序列时计算时间增加，可能导致 all-gather 的
   prefetch 策略失效，变成串行等待。

### 下一步行动

| 优先级 | 行动 | 预期效果 |
|--------|------|----------|
| P0 | 加 response truncation，限制 max_response_len 使 glen < 34000 | 直接消除 spike |
| P0 | 在 `HFModelWrapper.forward()` 里分 chunk 计算 logits/logprobs | 避免一次性分配 S×V 的 logits tensor |
| P1 | 加 sequence bucketing，减少 padding 浪费 | 减少 padding，间接降低 glen |
| P2 | ~~尝试 `npu_fusion_attention` 替代 SDPA~~ ❌ 不需要，SDPA 不是瓶颈且比 NFA 快 2x | - |
| P2 | 调查 `logprobs_from_logits` 的 `inplace_backward=True` 是否有效减少显存 | 可能已经在做 chunked 计算 |

### 分析脚本

- `codescout/scripts/analyze_spike.py` — 基础关联分析
- `codescout/scripts/analyze_spike_v2.py` — rank_0 vs entrypoint 对比
- `codescout/scripts/analyze_spike_v3.py` — DP 负载均衡验证
- `codescout/scripts/analyze_glen_time.py` — glen vs time 回归分析
- `codescout/scripts/analyze_cliff.py` — 性能悬崖精确定位
- `codescout/scripts/analyze_cliff_v2.py` — SDPA vs non-SDPA 分离分析
- `codescout/scripts/bench_sdpa_cliff.py` — 独立 SDPA/NFA benchmark（确认 SDPA 无悬崖）
- `codescout/scripts/bench_sdpa_cliff_v2.py` — 全模型 benchmark（OOM，需要 FSDP 多卡）

---

## 预期结论

| 结果 | 行动 |
|------|------|
| ~~Spike 跟序列长度强相关~~ ✅ 已确认 | 加 sequence truncation 或 bucketing |
| ~~SDPA 有性能悬崖~~ ❌ 独立 benchmark 无悬崖 | ~~尝试 npu_fusion_attention 或 triton-ascend FA~~ |
| ~~显存分配器频繁扩展~~ ✅ 间接确认（logits tensor 导致显存接近上限） | 分 chunk 计算 logits 或限制 glen |
| ~~HCCL 偶发卡顿~~ ❌ 已排除 (带宽稳定，通信只占 119ms) | ~~调整 HCCL timeout 参数或检查网络~~ |
| **lm_head logits tensor (S×V) 导致显存压力** ← 根因 | 分 chunk 计算或限制 glen < 34000 |

## 相关文件

- 训练 log: `/sharedata/liyuchen/workspace/logs/0401_1643_Qwen3-4B-Instruct-2507-npu-8x8.log`
- 训练 log (旧): `/tmp/ray/ray/session_*/logs/worker-*<skyrl_entrypoint_pid>*.err`
- SDPA benchmark: `SkyRL/npu_support/bench_attn_v2.py`
- 训练脚本: `codescout/scripts/run_async_training_npu.sh`
- FSDP strategy: `SkyRL/skyrl-train/skyrl_train/distributed/fsdp_strategy.py`
- Padding 逻辑: `SkyRL/skyrl-train/skyrl_train/dataset/preprocess.py` (`convert_prompts_responses_to_batch_tensors`)
