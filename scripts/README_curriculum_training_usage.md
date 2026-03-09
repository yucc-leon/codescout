# Curriculum Training 使用说明

本文记录 `scripts/run_curriculum_training.sh` 的常用启动方式和观察方法，避免遗忘。

## 1) 脚本能力概览

- 动态课程：`train -> stop -> refilter -> train`
- stage 切换保留优化器状态（checkpoint surgery）
- 外部控步：通过 `chunk` + 数据切片精确控制步数（不依赖框架 `max_steps`）
- 可选自适应早停（reward 饱和）
- WandB 可观察连续趋势：额外上报 `curriculum_global_step`

---

## 2) 基本启动

```bash
bash scripts/run_curriculum_training.sh \
  --raw-data data/adityasoni17__SWE-smith-py-code-search_train \
  --stages 2 \
  --max-steps-per-stage 100 \
  --output-dir output/curriculum_dynamic_2stage
```

---

## 3) 常用参数（建议先记这几个）

- `--max-steps-per-stage N`：每个 stage 最大训练步数
- `--chunk-steps N`：每个 chunk 训练步数（外部控步粒度）
- `--min-steps-per-stage N`：自适应早停最小步数
- `--adaptive-stop`：启用饱和早停
- `--saturation-patience N`：连续 N 个 chunk 无显著提升则停
- `--saturation-min-delta X`：显著提升阈值
- `--hf-save-interval auto|N`：HF 导出间隔
- `--ckpt-interval auto|N`：训练 checkpoint 间隔
- `--run-prefix PREFIX`：实验前缀（默认 `curriculum_dynctl_{N}stg`）
- `--run-suffix SUFFIX|none`：可选后缀（默认 `none`）
- `--prefilter-from DIR`：复用 prefilter 结果

---

## 4) 推荐模板

### A. 稳定版（不早停）

```bash
bash scripts/run_curriculum_training.sh \
  --raw-data data/adityasoni17__SWE-smith-py-code-search_train \
  --stages 2 \
  --max-steps-per-stage 100 \
  --chunk-steps 25 \
  --hf-save-interval auto \
  --ckpt-interval auto \
  --run-prefix curriculum_dynctl_2stg_v1 \
  --output-dir output/curriculum_dynctl_2stg_v1
```

### B. 自适应早停版

```bash
bash scripts/run_curriculum_training.sh \
  --raw-data data/adityasoni17__SWE-smith-py-code-search_train \
  --stages 2 \
  --max-steps-per-stage 100 \
  --chunk-steps 10 \
  --min-steps-per-stage 40 \
  --adaptive-stop \
  --saturation-patience 2 \
  --saturation-min-delta 0.01 \
  --run-prefix curriculum_dynctl_2stg_adapt \
  --output-dir output/curriculum_dynctl_2stg_adapt
```

### C. 复用 prefilter（加速重跑）

```bash
bash scripts/run_curriculum_training.sh \
  --raw-data data/adityasoni17__SWE-smith-py-code-search_train \
  --prefilter-from output/curriculum_dynctl_2stg_v1/prefilter \
  --stages 2 \
  --max-steps-per-stage 100 \
  --chunk-steps 25 \
  --run-prefix curriculum_dynctl_2stg_v1_retry \
  --output-dir output/curriculum_dynctl_2stg_v1_retry
```

---

## 5) WandB 上怎么看“连续趋势”

你不需要强制同一个 run id。现在脚本会给每个日志点附加：

- `curriculum_global_step = stage/chunk offset + local step`

在 WandB 图表中：

1. 打开任意指标图（如 reward / success）
2. 将 X 轴切换为 `curriculum_global_step`
3. 即可看到跨 stage/chunk 的连续趋势

---

## 6) 输出路径与命名

- 日志目录：`output/.../logs/<RUN_PREFIX>/`
- 进度文件：`output/.../progress_<RUN_PREFIX>.jsonl`
- stage ckpt：`${CKPT_BASE}/${RUN_PREFIX}_s<stage>/`
- stage 切换 patch init ckpt：`${CKPT_BASE}/${RUN_PREFIX}_s<stage>_init/`

---

## 7) 纯演练（不执行）

```bash
bash scripts/run_curriculum_training.sh \
  --raw-data data/adityasoni17__SWE-smith-py-code-search_train \
  --stages 2 \
  --max-steps-per-stage 100 \
  --chunk-steps 10 \
  --adaptive-stop \
  --dry-run
```

---

## 8) 手动 checkpoint patch（一般不用手动）

```bash
uv run python scripts/patch_checkpoint_for_stage.py \
  --src-ckpt-dir ckpts/<run>_s1/ \
  --dst-ckpt-dir ckpts/<run>_s2_init/
```

