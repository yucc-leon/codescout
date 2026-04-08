# 排查 NPU vs H200 精度/Reward 差距

## 现象

在 40k context、8x8 配置下，NPU 和 H200 的训练曲线存在系统性差距：

| 指标 | NPU (Ascend 910) | H200 | 差距 |
|------|-------------------|------|------|
| avg_final_rewards | ~1.5-2.0 | ~2.5-3.0 | **不可直接比较**（H200 额外加了 multiturn_reward） |
| multilevel_localization_f1_reward | 偏低 | 更高 | **核心异常**，纯定位准确率指标 |
| file_reward | 偏低 | 更高 | 异常 |
| module_reward | 偏低 | 更高 | 异常 |
| entity_reward | 偏低 | 更高 | 异常 |
| policy_loss | -0.3~-0.4 | ~-0.5 | 不同 |
| policy_entropy | 下降更快 | 下降更慢 | 收敛行为不同 |
| num_turns | ~4-5 | ~6-7 | agent 更早结束 |
| grad_norm | spike 更多 | 更平稳 | 训练不稳定 |

**注意**：`avg_final_rewards` 不可比，因为 H200 run 的 reward config 包含
`multiturn_reward`（4 turn 完成额外 +1.0），NPU run 当时没有。
真正反映问题的是 `multilevel_localization_f1_reward` — 这是纯定位准确率，
两边 config 一致，NPU 偏低说明模型在 NPU 上的定位能力确实更差。

## 可能原因

### 层级 1: 推理精度差异（影响 generation 质量）

vllm-ascend 0.11 在 NPU 上的推理输出可能跟 CUDA 版本有数值差异，
导致模型在相同 prompt 下生成不同的 token 序列。

验证方法：
```python
# 用同一个 prompt，分别在 NPU 和 H200 上跑 vLLM 推理
# 对比 logprobs 分布是否一致
# 如果 top-1 token 不同，说明推理精度有差异

import requests, json

prompt = "..."  # 用训练数据中的一个实际 prompt

# NPU 推理
resp_npu = requests.post("http://npu-host:8100/v1/completions", json={
    "model": "...", "prompt": prompt, "max_tokens": 50,
    "temperature": 0, "logprobs": 5
}).json()

# H200 推理
resp_h200 = requests.post("http://h200-host:8100/v1/completions", json={
    "model": "...", "prompt": prompt, "max_tokens": 50,
    "temperature": 0, "logprobs": 5
}).json()

# 对比 token 序列和 logprobs
```

如果 temperature=0 下两边输出不同，说明推理精度有差异。

### 层级 2: 训练精度差异（影响梯度更新）

SDPA vs Flash Attention 2 在 backward 时的梯度计算有数值差异。
FA2 用 online softmax 近似，SDPA 用标准实现。
两者在 fp32 下应该一致，但 bf16 下可能有差异。

验证方法：
```python
# 用同一个 batch 数据，分别在 NPU 和 H200 上跑一个 forward+backward
# 对比 loss 值和梯度 norm

# 1. 保存一个训练 batch 到文件
# （SkyRL 的 dump_data_batch=true 已经在做这个）

# 2. 在两个平台上加载同一个 batch，跑 forward+backward
# 3. 对比 loss 值（应该在 1e-3 以内）
# 4. 对比 gradient norm（应该在 1e-2 以内）
```

### 层级 3: bf16 matmul 精度差异

Ascend 910 的 bf16 矩阵乘法实现可能跟 H200 的 Tensor Core 有精度差异。
这会影响所有 linear 层的输出。

验证方法：
```python
import torch, torch_npu

# 在 NPU 上
a = torch.randn(1024, 1024, dtype=torch.bfloat16, device="npu")
b = torch.randn(1024, 1024, dtype=torch.bfloat16, device="npu")
c_npu = (a @ b).float().cpu()

# 在 H200 上（同样的 a, b）
# c_h200 = (a @ b).float().cpu()

# 对比 max abs diff 和 relative error
# diff = (c_npu - c_h200).abs()
# print(f"max diff: {diff.max()}, mean diff: {diff.mean()}")
```

### 层级 4: hermes tool parser 差异

之前发现 213 次 hermes tool call 解析失败。
如果 H200 上同样的模型解析失败率更低，说明推理输出格式有差异。

验证方法：
```bash
# 对比两个平台的 hermes parser 错误率
# NPU:
grep -c "Error in extracting tool call" $NPU_VLLM_LOG
# H200:
grep -c "Error in extracting tool call" $H200_VLLM_LOG
```

### 层级 5: weight sync 精度

HCCL broadcast 在传输模型权重时可能有精度损失。
每次 weight sync 后推理引擎的权重可能跟训练侧有微小差异。

验证方法：
```python
# 在 weight sync 前后，对比训练侧和推理侧的权重
# 取一个 layer 的权重做 allclose 检查
# torch.allclose(train_weight, infer_weight, atol=1e-6)
```

## 排查优先级

1. **推理精度对比**（最快，不需要重跑训练）
   - 用 base model（未训练）在两个平台上跑相同 prompt
   - temperature=0，对比输出 token 序列
   - 如果不同 → 推理精度是根因

2. **hermes parser 错误率对比**（从现有 log 提取）
   - 如果 NPU 错误率显著高于 H200 → 推理输出格式差异

3. **训练 loss 对比**（需要保存 batch 数据）
   - 用 dump_data_batch 保存的数据在两个平台跑
   - 对比 loss 值

4. **bf16 matmul 精度**（独立测试）
   - 如果 matmul 精度差异大 → 所有层都受影响

## 关联分析：num_turns 偏低

NPU 上 agent 平均 turn 数更少（4-5 vs 6-7），可能原因：
- 推理精度差异导致模型更早调用 finish tool（或生成 EOS）
- tool call 解析失败导致 conversation 提前终止
- 更短的对话 → 更少的信息收集 → 更低的 localization 准确率 → 更低的 reward

这跟 `generate/avg_tokens_non_zero_rewards` NPU 偏低一致 —
成功的 rollout 用了更少的 token，说明 agent 的搜索深度不够。

## 可能的修复方向

| 根因 | 修复 |
|------|------|
| 推理精度差异 | 尝试 fp32 推理（性能会降）或调整 vllm-ascend kernel |
| SDPA vs FA2 梯度差异 | 可接受，是硬件特性 |
| bf16 matmul 精度 | 尝试 fp32 accumulation（torch.set_float32_matmul_precision） |
| hermes parser 失败率高 | 优化 parser 或换用更鲁棒的 tool call 格式 |
| weight sync 精度损失 | 检查 HCCL broadcast 是否有精度选项 |

## 相关文件

- 训练 log: `/tmp/ray/ray/session_*/logs/`
- vLLM engine log: 同上，worker-*<vllm_pid>*.out
- dump_data_batch: `$CKPT_PATH/trajectories/`
- 训练脚本: `codescout/scripts/run_async_training_npu.sh`
