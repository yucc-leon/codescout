# SP=2 (配置 B) Reward 低于 SP=1 (配置 A) 的根因分析

> 日期: 2026-04-08

## 已排除的假设

| 假设 | 验证方法 | 结果 |
|------|----------|------|
| SDPA vs FA2 数值差异 | 同一输入对比 logprobs | diff=0，完全一致 |
| sample_packing 跨序列 attention 泄漏 | single vs packed 对比 | diff=0，隔离正确 |
| SP=2 position_ids 重建错误 | 模拟 slicing + all_gather | cu_seqlens 完全正确 |
| DP=2 vs DP=4 梯度方差 | 分析 FSDP all-reduce | 有效 batch size 相同（8 seqs） |

## 关键发现

### 1. NPU 上 padding 处理有问题（两个配置都受影响）

无论 SDPA 还是 FA2，batched 模式下 padding token 没有被正确 mask：
- Seq1 single vs batched: max logit diff = 5.25
- Seq2 single vs batched: max logit diff = 11.5

但 sample_packing 去掉了 padding，所以 packed 路径是正确的（diff=0）。

这意味着配置 A（无 packing）的 logprobs 被 padding 污染，配置 B（有 packing）的 logprobs 是干净的。
但由于 old_log_probs 和 log_probs 都用相同路径计算，log_ratio 中污染会抵消。

### 2. 配置 B 的 policy_loss 信号极弱

配置 B（SP=2）的训练日志显示：
- `policy_loss` 大部分 step 为 0.0 或 ~1e-9
- `ppo_clip_ratio` 始终为 0.0
- 偶尔出现非零 loss（-0.03 ~ -0.13），但很稀疏

配置 A（SP=1）的 policy_loss 更大且更频繁（-0.03 ~ -0.1）。

### 3. 根因推测：sample_packing 改变了 GSPO 的 sequence_mean reduction 行为

GSPO 使用 `sequence_mean` loss reduction：
```python
loss = masked_mean(loss, loss_mask, dim=-1).mean()
```

在非 packing 模式下，每个序列独立计算 masked_mean，padding 位置被 loss_mask 排除。

在 packing 模式下，所有序列被打包成 (1, nnz)。`num_actions` 用于从末尾 slice：
```python
action_log_probs = log_probs[:, -num_actions - 1 : -1]
```

但 packing 后 log_probs 的形状是 (1, nnz)，而 `num_actions` 是 batch 内最大的 response 长度。
这意味着 `action_log_probs` 的 slice 可能不正确 — 它从 packed 序列的末尾 slice，
而不是从每个序列的末尾 slice。

**等等** — 代码中 pad_input 已经把 log_probs 恢复到 (B, S) 形状了，
所以 num_actions slice 应该是正确的。

让我重新检查... 实际上 pad_input 在 num_actions slice 之前执行，
所以 action_log_probs 的 shape 是 (B, num_actions)，这是正确的。

### 4. 真正的根因：DP=2 下 GSPO 的有效梯度信号

虽然 FSDP all-reduce 后有效 batch size 相同（8 seqs），但 GSPO 的
`sequence_mean` reduction 在每个 DP rank 上独立计算 loss，然后 all-reduce 梯度。

DP=4 时：每个 rank 处理 2 个 sequences，loss = mean of 2 per-seq losses
DP=2 时：每个 rank 处理 4 个 sequences，loss = mean of 4 per-seq losses

all-reduce 后梯度是等价的。所以这也不是原因。

### 5. 最可能的根因：micro_train_batch_size 与 packing 的交互

`micro_train_batch_size_per_gpu=1`。在非 packing 模式下，每个 micro-batch 是 1 个序列。
在 packing 模式下，每个 micro-batch 也是 1 个序列，但这个 "序列" 是打包后的 (1, nnz)，
包含了多个原始序列。

这意味着 packing 模式下，gradient accumulation 的行为不同：
- 非 packing：accumulate 8 个 micro-batches（每个 1 seq），然后 step
- packing：accumulate 8 个 micro-batches（每个 1 packed seq = 多个原始 seq），然后 step

但 `policy_mini_batch_size=8`，`micro_train_batch_size_per_gpu=1`，
所以 accumulation_steps = 8/1 = 8（非 packing）或 8/1 = 8（packing）。

在 packing 模式下，每个 micro-batch 的 (1, nnz) 只包含 1 个原始序列
（因为 BatchIterator 按 sample_batch_size=1 切分）。
所以 packing 在 micro_batch_size=1 时实际上没有打包多个序列。

**这就是问题所在**：micro_batch_size=1 时，packing 只是去掉了单个序列的 padding，
不会把多个序列打包在一起。但 SP=2 要求 use_sample_packing=true，
而 packing 改变了 attention 路径（不传 attention_mask）。

## 待验证

1. 确认 BatchIterator 在 packing 模式下每个 micro-batch 包含几个序列
2. 在 H200 上跑相同的 SP=2 配置，确认是否也有 reward 下降
3. 尝试增大 micro_train_batch_size_per_gpu 看是否改善

## 更新的分析 (2026-04-08 晚)

### 实验结果

1. SDPA vs FA2 在 NPU 上 logprobs 完全一致 (diff=0)
2. sample_packing 序列隔离正确 (single vs packed diff=0)
3. NPU 上 batched 模式 padding 处理有问题 (max logit diff=11.5)，但两个配置都受影响
4. micro_train_batch_size=1 时 packing 只去掉单序列 padding，不打包多序列
5. 配置 B 的 policy_loss 大部分为 0，clip_ratio 始终为 0
6. 配置 B 的 entropy 比配置 A 低 3-5 倍 (0.04 vs 0.15)
7. `FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False`，两个配置都走 `logprobs_from_logits_v2`
8. Standard vs Packed 路径的 action_log_probs 有 ~0.0003 mean diff（来自 padding 污染），
   但两个配置内部都是自洽的

## 根因定位 (2026-04-08 深夜)

### 实验结果

逐层测试 B=1 vs B=2（同一输入，无 padding）的数值差异：

| 组件 | max diff | 状态 |
|------|----------|------|
| npu_fusion_attention kernel | 0.000000 | OK |
| RMSNorm | 0.000000 | OK |
| nn.Linear (bf16) | 0.007812 | **BUG** |
| Embedding | 0.000000 | OK |
| 单层 transformer | 0.000000 | OK |
| 完整模型 (36层) | 4.820312 | **BUG** |

nn.Linear 在 NPU 上 B=1 vs B=2 有 0.0078 的 diff。单层看不出来（被其他精确的 op 稀释），
但 36 层累积后放大到 4.82。

完整模型的 logprob diff: mean=0.0047, **26.9% 的 token 超过 GSPO 阈值 (0.0003)**。

### 根因

`torch_npu` 的 bf16 matmul（nn.Linear 底层）在不同 batch size 下使用了不同的 tiling/累加策略，
导致浮点累加顺序不同，产生数值差异。这是 CANN/昇腾硬件层面的问题。

H200 上 `flash_attn=true`（默认值），transformers 的 FA2 路径会用 `_upad_input` 在 attention
之前物理去掉 padding，所以 padding 不影响计算。而且 CUDA 上 nn.Linear 的 bf16 matmul
在不同 batch size 下是 bit-exact 的（或差异极小）。

NPU 配置 A 用 `flash_attn=false`（SDPA），padding 靠 attention_mask 做 softmax masking。
但由于 nn.Linear 的 batch-dependent 行为，即使 attention 正确 mask 了 padding，
MLP 层的输出已经因为 batch 中其他序列的存在而改变了。

### 影响

1. **配置 A vs H200**：NPU 上 nn.Linear 的 batch-dependent 行为导致每个 training step
   的 logprobs 和 H200 有系统性偏差。虽然 log_ratio 中偏差会部分抵消（同一 batch 的
   两次 forward 偏差一致），但梯度方向的累积偏差导致模型学到不同的策略。

2. **配置 B vs 配置 A**：配置 B 用了 sample_packing（去掉 padding 后 B=1），
   配置 A 不用 packing（B=batch_size，有 padding）。两者走了不同的 nn.Linear 计算路径，
   产生不同的数值结果，导致不同的训练动态。

## 建议的下一步

1. **最高优先级：让同事在 H200 上跑 nn.Linear B=1 vs B=2 测试**
   - 如果 H200 上 diff=0 → 确认是 NPU/CANN 特有问题
   - 脚本：`codescout/scripts/quantify_batch_dim_bug.py`（去掉 npu_support import 即可在 CUDA 上跑）

2. **尝试 fp32 matmul accumulation**
   - `torch.set_float32_matmul_precision('highest')` 或 CANN 等效设置
   - 如果 nn.Linear 的 diff 消失 → 确认是 bf16 累加精度问题

3. **配置 A workaround：用 flash_attn=true（跟 H200 一致）**
   - H200 默认 flash_attn=true，NPU 配置 A 改成 false 是因为当时没有 flash_attn
   - transformers 4.57 在 NPU 上已有 npu_flash_attn_func/varlen_func
   - flash_attn=true 时有 padding 的 batch 走 varlen 路径（unpad → attention → pad）

4. **配置 B 的 policy_loss 更小的问题需要在 nn.Linear bug 解决后重新评估**

