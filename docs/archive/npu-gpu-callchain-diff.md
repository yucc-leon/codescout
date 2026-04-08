# NPU vs GPU 训练调用链路差异分析

## 差异点总览

从 `patch_cuda.py` 和训练配置出发，NPU 和 GPU 的训练链路有以下差异：

### 差异 1: flash_attn.ops.triton.cross_entropy 被 stub

**GPU 链路：**
```
logprobs_from_logits()
  → FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
  → logprobs_from_logits_flash_attn(logits, labels)
    → flash_attn.ops.triton.cross_entropy.cross_entropy_loss(logits, labels, inplace_backward=True)
    → 返回 -output[0]  (fused CUDA kernel, 内部做 log_softmax + gather)
```

**NPU 链路：**
```
logprobs_from_logits()
  → FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False  ← stub 导致
  → logprobs_from_logits_v2(logits, labels)
    → for row in logits:
        F.log_softmax(row.float(), dim=-1)  ← fp32 upcast
        .gather(-1, labels)
    → 返回 logprobs
```

**影响分析：**
- 数值差异：我们测试过，fp32 log_softmax 和 F.cross_entropy 的结果完全一致（diff=0）
- 但 flash_attn 的 cross_entropy_loss 是 fused CUDA kernel，其内部精度可能和 PyTorch 的
  F.log_softmax 不完全一致（特别是 logsumexp 的计算顺序）
- **关键问题：flash_attn 的 cross_entropy_loss 是否也做了 fp32 upcast？**
  如果 flash_attn 在 bf16 下直接计算（不 upcast），而 NPU 的 _v2 做了 fp32 upcast，
  那两边的 logprobs 精度不同。但这应该让 NPU 更精确，不是更差。
- **inplace_backward=True 的影响：** flash_attn 的 cross_entropy 支持 inplace backward，
  这意味着 backward 时直接修改 logits tensor 而不是分配新的 gradient tensor。
  NPU 的 _v2 没有这个优化，backward 时会分配额外的 gradient tensor。
  这不影响数值，但影响显存。

**结论：这个差异不太可能导致训练不稳定。两边的 logprobs 数值应该一致。**

### 差异 2: flash_attn=false vs flash_attn=true (attn_implementation)

**GPU 链路（flash_attn=true，默认值）：**
```
HFModelWrapper.forward()
  → self.model(sequences, attention_mask=attn_mask, position_ids=pos_ids)
    → Qwen3ForCausalLM.forward()
      → Qwen3Model.forward()
        → 每层 Qwen3DecoderLayer:
          → Qwen3Attention.forward()
            → config._attn_implementation = "flash_attention_2"
            → ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
            → _flash_attention_forward()
              → 在 NPU 上：最终调 torch.nn.functional.scaled_dot_product_attention
                          或 torch_npu.npu_fusion_attention
```

**NPU 链路（flash_attn=false）：**
```
HFModelWrapper.forward()
  → self.model(sequences, attention_mask=attn_mask, position_ids=pos_ids)
    → Qwen3ForCausalLM.forward()
      → Qwen3Model.forward()
        → 每层 Qwen3DecoderLayer:
          → Qwen3Attention.forward()
            → config._attn_implementation = "sdpa"
            → ALL_ATTENTION_FUNCTIONS["sdpa"]
            → _sdpa_attention_forward()
              → torch.nn.functional.scaled_dot_product_attention
                → 在 NPU 上：最终调 npu_fusion_attention
```

**影响分析：**
- 底层 kernel 相同（都是 npu_fusion_attention / SDPA）
- 但 HuggingFace 的 flash_attention_2 和 sdpa 路径在 attention mask 处理上不同：
  - flash_attention_2: 可能不传 attention_mask（用 position_ids 代替）
  - sdpa: 传 attention_mask，由 SDPA 内部处理
- 我们测试过：非 padding 位置的 logits 完全一致，只有 padding 位置不同
- padding 位置被 loss_mask 排除，不影响 loss

**结论：这个差异不影响训练数值。**

### 差异 3: torch.cuda → torch.npu 代理

**patch_cuda.py 的代理机制：**
```python
sys.modules["torch.cuda"] = _CudaProxy(orig_cuda, npu)
```

代理的 `__getattr__` 会把所有未显式覆盖的属性转发到 `torch.npu`。
显式覆盖的有：is_available, device_count, set_device, empty_cache, synchronize,
manual_seed, manual_seed_all, memory_allocated, max_memory_allocated, get_rng_state,
set_rng_state, mem_get_info, get_device_properties, ipc_collect

**潜在问题：**
- `torch.cuda.amp` 相关的函数没有显式覆盖。`torch.autocast(device_type="cuda")`
  在 NPU 上会走代理，最终调 `torch.npu` 的 autocast。但 autocast 的 op 覆盖列表
  可能不同——NPU 的 autocast 可能对某些 op 不做 bf16 降精度，或者对某些 op 做了
  GPU 上不做的降精度。
- **这是一个值得深入调查的方向。**

### 差异 4: NCCL → HCCL

```python
torch.distributed.init_process_group: "nccl" → "hccl"
torch.distributed.new_group: "nccl" → "hccl"
```

**影响分析：**
- 我们测试过 HCCL 的 all-reduce 和 reduce-scatter 精度，完全一致
- 但 HCCL 和 NCCL 的 **reduce 顺序** 可能不同。对于 fp32 reduce，
  浮点加法的结合律不成立，不同的 reduce 顺序会产生微小差异
- 这个差异在单次 reduce 中极小（~1e-7），但经过几百个 step 的累积...
- **但我们测试过 100 步 optimizer step 的累积，参数 bit-exact。**
  所以 reduce 顺序差异不是问题。

### 差异 5: Tensor.cuda() → Tensor.npu()

```python
torch.Tensor.cuda = _tensor_cuda  # 重定向到 .npu()
```

**影响分析：**
- 这只影响显式调用 `.cuda()` 的代码
- SkyRL 的训练代码主要用 `torch.cuda.current_device()` 和 `.to(device)`
- 不太可能有影响

### 差异 6: DeviceMesh "cuda" → "npu"

```python
init_device_mesh("cuda", ...) → init_device_mesh("npu", ...)
```

**影响分析：**
- FSDP2 用 DeviceMesh 管理分片。device_type 从 "cuda" 变成 "npu"
- 这应该只影响设备类型标识，不影响计算
- 但 FSDP2 内部可能有基于 device_type 的分支逻辑...

## 最可疑的方向

### 方向 A: autocast 的 op 覆盖列表差异

`torch.autocast(dtype=torch.bfloat16, device_type="cuda")` 在 NPU 上通过代理
最终变成 NPU 的 autocast。但 NPU 和 GPU 的 autocast op 列表可能不同。

如果某个关键 op（比如 `torch.mm`、`F.linear`、`torch.bmm`）在 GPU 上被 autocast
降到 bf16，但在 NPU 上保持 fp32（或反过来），就会导致中间结果的精度不同。

特别是在 GSPO 的 loss 计算链路中：
```
log_probs (fp32) - old_log_probs (fp32) → log_ratio (fp32)
masked_mean(log_ratio, ...) → log_importance_weights (fp32?)
log_probs - log_probs.detach() + log_importance_weights.detach() → log_token_importance_weights
torch.exp(log_token_importance_weights) → ratio
ratio * advantages → surr1
ratio.clamp(...) * advantages → surr2
torch.min(surr1, surr2) → loss
```

如果 `advantages` 在 autocast 下被降到 bf16，而 `ratio` 是 fp32，
混合精度的行为可能不同。

### 方向 B: torch_npu 的某些 op 实现有数值差异

虽然单次调用是确定性的，但某些 op 在 NPU 上的实现可能和 GPU 有系统性偏差。
比如 `torch.exp()`、`torch.clamp()`、`masked_mean()` 等。
这些偏差在单次调用中极小，但在 RL 训练的反馈循环中可能被放大。

### 方向 C: 随机数生成器差异

`torch.npu.manual_seed` 和 `torch.cuda.manual_seed` 的 RNG 实现不同。
虽然训练脚本没有显式设置 seed，但 dropout、数据 shuffle 等操作依赖 RNG。
如果 NPU 的 RNG 分布和 GPU 不同，可能导致训练轨迹不同。
但这不能解释"NPU 更不稳定"——不同的随机种子应该只是不同的轨迹，不是更差的轨迹。

## 推荐的下一步验证

1. **验证 autocast op 列表差异：** 在 NPU 和 GPU 上分别打印 autocast 的 op 列表，
   对比差异。
2. **在 GSPO loss 计算链路中逐 op 检查 dtype：** 在 training_step 里加 print，
   检查 log_ratio、advantages、ratio、surr1、surr2 的 dtype 和数值范围。
3. **对比 torch_npu 和 torch 的基础 op 精度：** 对 exp、clamp、softmax 等
   在相同输入下对比 NPU 和 CPU 的输出。
