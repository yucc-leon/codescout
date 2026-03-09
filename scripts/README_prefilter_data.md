# prefilter_data.py 使用说明

## 概述

`prefilter_data.py` 用于在训练前对数据进行**轻量级离线 rollout**，并根据 reward 过滤、排序，得到课程式（由易到难）的过滤后数据集，供后续 RL 训练使用。

### 主要能力

- **Rollout + 过滤**：对每条 instance 做多轮 agent 推理（terminal + localization_finish），用配置中的 reward 函数打分，再按阈值过滤并保存
- **仅 Rollout**（`--rollout-only`）：只跑 rollout，结果写到 `output/rollout_results.jsonl`，不做过滤与保存 parquet
- **仅过滤**（`--filter-only`）：基于已有 rollout 结果目录，按 `--min-reward` 过滤并生成 train/validation parquet
- **N 样本取最优**（`--n-samples`）：每个 instance 跑 N 次 rollout，保留 reward 最高的一条
- **数据并行**（`--dp-size`）：多 GPU 多进程并行 rollout，加快吞吐

---

## 环境与依赖

- Python 3，需安装：`vllm`、`pandas`、`omegaconf`、`jinja2`、`tqdm`
- 项目结构需包含：`configs/rewards/*.yaml`、`src/prompts/`、`src/rewards`、`src/utils/instance`
- 可选环境变量：
  - `MODEL_PATH_4B`、`MODEL_PATH_14B`：覆盖默认 4B/14B 模型路径

---

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | - | 模型规模，`4b` 或 `14b`（filter-only 时可省略） |
| `--input` | str | 自动检测 | 输入数据集路径，支持 `.parquet` 或 `.jsonl` |
| `--output` | str | **必填** | 输出目录（rollout 结果与过滤后数据均在此） |
| `--config` | str | `configs/rewards/baseline_4b.yaml` | 实验配置（需含 reward/tools/prompts） |
| `--rollout-only` | flag | - | 只跑 rollout，不写 train/validation parquet |
| `--filter-only` | flag | - | 仅做过滤，需配合 `--rollout-dir` |
| `--rollout-dir` | str | - | 已有 rollout 结果目录（用于 `--filter-only`） |
| `--min-reward` | float | 0.0 | 过滤时保留 reward 大于该值的样本 |
| `--max-samples` | int | None | 最多处理的 instance 数量（用于调试或子集） |
| `--temperature` | float | 0.6 | 采样温度 |
| `--max-turns` | int | 10 | 单条 instance 最大对话轮数 |
| `--tp-size` | int | 按 model 默认 | 张量并行大小 |
| `--gpu-memory-utilization` | float | 0.85 | vLLM GPU 显存占用比例 |
| `--dry-run` | flag | - | 只打印将要保存的样本数，不写文件 |
| `--resume` | flag | - | 从已有 `rollout_results.jsonl` 续跑，跳过已完成 instance |
| `--n-samples` | int | 1 | 每个 instance 的 rollout 次数，取 reward 最高的一条 |
| `--dp-size` | int | 自动 | 数据并行 worker 数（多进程多 GPU） |
| `--visible-gpus` | str | None | 可见 GPU 列表，如 `0,1,2,3` |
| `--shuffle` | flag | - | 打乱输入 instance 顺序 |
| `--seed` | int | 42 | 随机种子（与 `--shuffle` 配合） |
| `--checkpoint` | str | None | 使用指定 checkpoint 路径作为模型 |

---

## 使用方式与示例

### 1. 完整流程：Rollout + 过滤并生成 train/validation

使用 4B 模型、默认 config，输出到 `out/prefilter_baseline`，只处理前 100 条做快速验证：

```bash
cd /path/to/acs_soni_main

python scripts/prefilter_data.py \
  --model 4b \
  --output out/prefilter_baseline \
  --max-samples 100
```

不指定 `--input` 时，脚本会按 `DATASET_PATHS` 自动查找项目内的 parquet（如 `data/swe_smith/train.parquet`）。

指定输入与配置、提高过滤阈值：

```bash
python scripts/prefilter_data.py \
  --model 4b \
  --input data/swe_smith/train.parquet \
  --output out/prefilter_strict \
  --config configs/rewards/baseline_4b.yaml \
  --min-reward 0.5
```

### 2. 仅 Rollout，不生成过滤后的数据集

适合先大规模跑完 rollout，之后再多次调 `--min-reward` 做过滤：

```bash
python scripts/prefilter_data.py \
  --model 4b \
  --input data/swe_smith/train.parquet \
  --output out/rollout_only \
  --rollout-only
```

结果在 `out/rollout_only/rollout_results.jsonl`，每行一个 JSON（含 `instance_id`、`reward`、`reward_details`、`num_turns` 等）。

### 3. 仅过滤：从已有 Rollout 结果生成 train/validation

在已有 rollout 目录上，只做按 reward 过滤和排序，并写入 train/validation parquet：

```bash
python scripts/prefilter_data.py \
  --filter-only \
  --rollout-dir out/rollout_only \
  --input data/swe_smith/train.parquet \
  --output out/filtered_v2 \
  --min-reward 0.3 \
  --max-samples 5000
```

`--input` 需与当时做 rollout 时用的数据集一致（用于对齐列和保存 `prefilter_reward`）。

### 4. 断点续跑（Resume）

第一次跑了一部分后中断，可用 `--resume` 从 `rollout_results.jsonl` 中已完成的 `instance_id` 继续：

```bash
python scripts/prefilter_data.py \
  --model 4b \
  --input data/swe_smith/train.parquet \
  --output out/prefilter_baseline \
  --resume
```

### 5. 多 GPU 数据并行加速

使用 4 张卡做数据并行（每卡一个 4B 进程），每个 instance 跑 2 次取最优：

```bash
python scripts/prefilter_data.py \
  --model 4b \
  --input data/swe_smith/train.parquet \
  --output out/prefilter_dp \
  --dp-size 4 \
  --visible-gpus 0,1,2,3 \
  --n-samples 2
```

14B 模型通常 `tp_size=2`，会按 `tp_size` 占用 GPU，`dp-size` 为实际并行 worker 数（总 GPU 数 / tp_size）。

### 6. 使用自定义 checkpoint

不改用默认 4B/14B 路径，而是指定训练得到的 checkpoint：

```bash
python scripts/prefilter_data.py \
  --model 4b \
  --checkpoint /path/to/your/checkpoint-1000 \
  --input data/swe_smith/train.parquet \
  --output out/prefilter_ckpt
```

### 7. Dry-run：只查看过滤后数量

不写任何文件，只打印“会保留多少条”：

```bash
# 先有 rollout 结果
python scripts/prefilter_data.py \
  --filter-only \
  --rollout-dir out/rollout_only \
  --input data/swe_smith/train.parquet \
  --output out/dry \
  --min-reward 0.5 \
  --dry-run
```

### 8. 打乱顺序并限制数量（便于做子集实验）

```bash
python scripts/prefilter_data.py \
  --model 4b \
  --input data/swe_smith/train.parquet \
  --output out/prefilter_shuffle \
  --shuffle \
  --seed 123 \
  --max-samples 500
```

---

## 输出说明

- **Rollout 阶段**（未加 `--filter-only` 时）  
  - `{output}/rollout_results.jsonl`：每行一个 rollout 结果（含 `instance_id`、`reward`、`reward_details`、`num_turns`、`error` 等）。

- **过滤并写数据集**（默认或 `--filter-only` 且未 `--dry-run`）  
  - `{output}/train.parquet`、`{output}/validation.parquet`：按 reward 从高到低排序，并带有 `prefilter_reward` 列；默认约 95% train、5% validation。  
  - `{output}/stats.json`：统计信息（如 `total_rollouts`、`filtered_samples`、`avg_reward`、`min_reward`、`max_reward` 等）。

---

## 常见问题

1. **Dataset not found**  
   确保项目根目录下存在 `data/swe_smith/train.parquet` 或 `data/adityasoni17__SWE-smith-py-code-search_train/train.parquet`，或用 `--input` 显式指定路径。

2. **`--filter-only` 报错**  
   必须同时指定 `--rollout-dir`，且该目录下存在 `rollout_results.jsonl`。

3. **仅做过滤时要不要 `--model`**  
   不需要，`--filter-only` 时不加载模型。

4. **多卡时 OOM**  
   可适当调低 `--gpu-memory-utilization`（如 0.7）或减小 `--tp-size`（仅 14B）、减少 `--dp-size`。

以上为 `prefilter_data.py` 的使用说明与示例，按需组合参数即可复现和扩展你的 prefilter 流程。
