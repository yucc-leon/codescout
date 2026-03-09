#!/usr/bin/env python3
"""
Analyze curriculum training diagnostics from local artifacts.

Outputs:
  - diagnostics.json (structured metrics)
  - diagnostics.md   (human-readable summary)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BUCKETS = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.01]
BUCKET_LABELS = ["[0,0.25)", "[0.25,0.5)", "[0.5,1.0)", "[1.0,1.5)", "[1.5,2.0)", "[2.0,3.0]"]


@dataclass
class MarkdownRow:
    experiment: str
    step: str
    file_f1: float
    module_f1: float
    entity_f1: float
    success_rate: str
    avg_steps: float
    avg_tools: float
    output_format: str

    @property
    def score_mean_f1(self) -> float:
        return (self.file_f1 + self.module_f1 + self.entity_f1) / 3.0

    @property
    def step_num(self) -> int | None:
        if re.fullmatch(r"\d+", self.step):
            return int(self.step)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze curriculum diagnostics from logs/data/results")
    parser.add_argument("--curriculum-dir", required=True, help="Curriculum output directory")
    parser.add_argument("--baseline-log", required=True, help="Baseline training log path")
    parser.add_argument("--results-md", required=True, help="Markdown results table path")
    parser.add_argument(
        "--baseline-exp",
        default="Baseline-4B-2507-sonibugfix-withrg",
        help="Baseline experiment name in results markdown",
    )
    parser.add_argument(
        "--curriculum-prefix",
        default="curriculum_dynamic_2stg",
        help="Curriculum experiment name prefix in results markdown",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for diagnostics.{json,md} (default: <curriculum-dir>/analysis)",
    )
    return parser.parse_args()


def read_table_rows(md_path: Path) -> list[MarkdownRow]:
    rows: list[MarkdownRow] = []
    lines = md_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        if not line.startswith("|"):
            continue
        if line.startswith("|---") or line.startswith("| Experiment "):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 9:
            continue
        try:
            rows.append(
                MarkdownRow(
                    experiment=parts[0],
                    step=parts[1],
                    file_f1=float(parts[2]),
                    module_f1=float(parts[3]),
                    entity_f1=float(parts[4]),
                    success_rate=parts[5],
                    avg_steps=float(parts[6]),
                    avg_tools=float(parts[7]),
                    output_format=parts[8],
                )
            )
        except ValueError:
            continue
    return rows


def summarize_rows(rows: list[MarkdownRow]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    best_by_mean = max(rows, key=lambda r: r.score_mean_f1)
    best_by_module = max(rows, key=lambda r: r.module_f1)
    by_step = {}
    for r in rows:
        if r.step_num is None:
            continue
        by_step[r.step_num] = {
            "file_f1": r.file_f1,
            "module_f1": r.module_f1,
            "entity_f1": r.entity_f1,
            "mean_f1": r.score_mean_f1,
            "avg_steps": r.avg_steps,
            "avg_tools": r.avg_tools,
            "success_rate": r.success_rate,
        }
    return {
        "count": len(rows),
        "best_by_mean_f1": {
            "experiment": best_by_mean.experiment,
            "step": best_by_mean.step,
            "file_f1": best_by_mean.file_f1,
            "module_f1": best_by_mean.module_f1,
            "entity_f1": best_by_mean.entity_f1,
            "mean_f1": best_by_mean.score_mean_f1,
        },
        "best_by_module_f1": {
            "experiment": best_by_module.experiment,
            "step": best_by_module.step,
            "file_f1": best_by_module.file_f1,
            "module_f1": best_by_module.module_f1,
            "entity_f1": best_by_module.entity_f1,
            "mean_f1": best_by_module.score_mean_f1,
        },
        "by_step": dict(sorted(by_step.items(), key=lambda x: x[0])),
    }


def reward_col(df: pd.DataFrame) -> str | None:
    for col in ("prefilter_reward", "reward"):
        if col in df.columns:
            return col
    return None


def distribution_of_parquet(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    col = reward_col(df)
    if not col:
        return {"path": str(path), "error": f"no reward column in {list(df.columns)}"}
    series = df[col].astype(float)
    cats = pd.cut(series, bins=BUCKETS, labels=BUCKET_LABELS, include_lowest=True, right=False)
    bucket_cnt = cats.value_counts().reindex(BUCKET_LABELS, fill_value=0)
    return {
        "path": str(path),
        "reward_col": col,
        "n": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
        "q10": float(series.quantile(0.1)),
        "q50": float(series.quantile(0.5)),
        "q90": float(series.quantile(0.9)),
        "bucket_ratio": {k: float(v / len(series)) for k, v in bucket_cnt.items()},
    }


def load_ids(path: Path) -> pd.Series:
    return pd.read_parquet(path)["instance_id"].astype(str)


def stage_overlap(stage_files: dict[str, Path]) -> dict[str, Any]:
    keys = sorted(stage_files.keys(), key=lambda s: int(re.sub(r"[^\d]", "", s)))
    id_sets = {k: set(load_ids(stage_files[k]).tolist()) for k in keys}
    out: dict[str, Any] = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            inter = len(id_sets[a] & id_sets[b])
            union = len(id_sets[a] | id_sets[b])
            out[f"{a} vs {b}"] = {
                "intersection": inter,
                "jaccard": (inter / union) if union else 0.0,
                f"recall_{a}": inter / len(id_sets[a]) if id_sets[a] else 0.0,
                f"recall_{b}": inter / len(id_sets[b]) if id_sets[b] else 0.0,
            }
    return out


def rollout_stats(jsonl_path: Path) -> dict[str, Any]:
    count = 0
    called_finish = 0
    rewards = []
    turns = []
    errors = 0
    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
            if row.get("called_finish_tool"):
                called_finish += 1
            if row.get("error") is not None:
                errors += 1
            reward = row.get("reward")
            if isinstance(reward, (int, float)):
                rewards.append(float(reward))
            num_turns = row.get("num_turns")
            if isinstance(num_turns, int):
                turns.append(num_turns)

    out: dict[str, Any] = {"path": str(jsonl_path), "count": count}
    if count == 0:
        return out
    out["finish_ratio"] = called_finish / count
    out["error_ratio"] = errors / count
    if rewards:
        s = pd.Series(rewards, dtype=float)
        out["reward"] = {
            "mean": float(s.mean()),
            "q10": float(s.quantile(0.1)),
            "q50": float(s.quantile(0.5)),
            "q90": float(s.quantile(0.9)),
        }
    if turns:
        t = pd.Series(turns, dtype=float)
        out["turns"] = {
            "mean": float(t.mean()),
            "q10": float(t.quantile(0.1)),
            "q50": float(t.quantile(0.5)),
            "q90": float(t.quantile(0.9)),
        }
    return out


def log_failure_counts(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "oom_or_mem": r"out of memory|Free memory on device|CUDA out of memory|less than desired GPU memory utilization",
        "engine_init_failed": r"Engine core initialization failed|Exception raised in creation task",
        "port_conflict": r"address already in use|Address already in use|Errno 98",
        "nccl_warn": r"NCCL|destroy_process_group\(\) was not called",
        "traceback": r"Traceback \(most recent call last\)",
    }
    counts = {k: len(re.findall(v, text, flags=re.IGNORECASE)) for k, v in patterns.items()}
    return {"path": str(log_path), "line_count": text.count("\n") + 1, "counts": counts}


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Curriculum Diagnostics")
    lines.append("")
    lines.append("## Eval Results Summary")
    lines.append("")
    base = report["results"]["baseline"]
    curr = report["results"]["curriculum"]
    lines.append(f"- Baseline rows: {base.get('count', 0)}")
    lines.append(f"- Curriculum rows: {curr.get('count', 0)}")
    if base.get("count", 0) > 0:
        b = base["best_by_module_f1"]
        lines.append(
            f"- Baseline best module F1: step {b['step']} | file/module/entity = "
            f"{b['file_f1']:.4f}/{b['module_f1']:.4f}/{b['entity_f1']:.4f}"
        )
    if curr.get("count", 0) > 0:
        c = curr["best_by_module_f1"]
        lines.append(
            f"- Curriculum best module F1: step {c['step']} | file/module/entity = "
            f"{c['file_f1']:.4f}/{c['module_f1']:.4f}/{c['entity_f1']:.4f}"
        )
    lines.append("")
    lines.append("## Reward Distribution")
    lines.append("")
    for name, dist in report["reward_distribution"].items():
        if "error" in dist:
            lines.append(f"- {name}: {dist['error']}")
            continue
        lines.append(
            f"- {name}: n={dist['n']}, mean={dist['mean']:.3f}, "
            f"q10/q50/q90={dist['q10']:.3f}/{dist['q50']:.3f}/{dist['q90']:.3f}"
        )
    lines.append("")
    lines.append("## Rollout Behavior")
    lines.append("")
    for name, stats in report["rollout"].items():
        if stats.get("count", 0) == 0:
            lines.append(f"- {name}: empty")
            continue
        r = stats.get("reward", {})
        t = stats.get("turns", {})
        lines.append(
            f"- {name}: count={stats['count']}, finish_ratio={stats.get('finish_ratio', 0):.3f}, "
            f"turns_mean={t.get('mean', 0):.2f}, reward_mean={r.get('mean', 0):.3f}"
        )
    lines.append("")
    lines.append("## Log Failures")
    lines.append("")
    for name, info in report["log_failures"].items():
        c = info["counts"]
        lines.append(
            f"- {name}: oom_or_mem={c['oom_or_mem']}, engine_init_failed={c['engine_init_failed']}, "
            f"port_conflict={c['port_conflict']}, nccl_warn={c['nccl_warn']}, traceback={c['traceback']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    curriculum_dir = Path(args.curriculum_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else curriculum_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Parse eval table
    rows = read_table_rows(Path(args.results_md).resolve())
    base_rows = [r for r in rows if r.experiment == args.baseline_exp]
    cur_rows = [r for r in rows if r.experiment.startswith(args.curriculum_prefix)]

    # 2) Reward distributions from parquet artifacts
    reward_distribution: dict[str, Any] = {}
    candidate_files: dict[str, Path] = {}
    prefilter = curriculum_dir / "prefilter" / "train.parquet"
    if prefilter.exists():
        candidate_files["prefilter"] = prefilter
    for p in sorted(curriculum_dir.glob("refilter_stage*/train.parquet")):
        candidate_files[p.parent.name] = p
    for p in sorted(curriculum_dir.glob("stage_data/stage*/train.parquet")):
        candidate_files[p.parent.name] = p
    for name, path in candidate_files.items():
        reward_distribution[name] = distribution_of_parquet(path)

    # 3) Overlap across stage data
    stage_files = {k: v for k, v in candidate_files.items() if k.startswith("stage")}
    overlap = stage_overlap(stage_files) if len(stage_files) >= 2 else {}

    # 4) Rollout diagnostics
    rollout: dict[str, Any] = {}
    for p in sorted(curriculum_dir.glob("**/rollout_results.jsonl")):
        rel_name = str(p.parent.relative_to(curriculum_dir))
        rollout[rel_name] = rollout_stats(p)

    # 5) Log failure diagnostics
    log_failures = {
        "baseline": log_failure_counts(Path(args.baseline_log).resolve()),
    }
    for p in sorted((curriculum_dir / "logs").glob("train_stage*.log")):
        log_failures[p.name] = log_failure_counts(p)

    report = {
        "inputs": {
            "curriculum_dir": str(curriculum_dir),
            "baseline_log": str(Path(args.baseline_log).resolve()),
            "results_md": str(Path(args.results_md).resolve()),
            "baseline_exp": args.baseline_exp,
            "curriculum_prefix": args.curriculum_prefix,
        },
        "results": {
            "baseline": summarize_rows(base_rows),
            "curriculum": summarize_rows(cur_rows),
        },
        "reward_distribution": reward_distribution,
        "stage_overlap": overlap,
        "rollout": rollout,
        "log_failures": log_failures,
    }

    json_path = out_dir / "diagnostics.json"
    md_path = out_dir / "diagnostics.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
