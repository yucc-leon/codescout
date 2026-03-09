import json, os, sys
rollout_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ROLLOUT_JSONL")
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("OUTPUT_DIR")
if not rollout_path or not os.path.isfile(rollout_path):
    print("Usage: python write_eval_summary.py <rollout.jsonl> [output_dir]", file=sys.stderr)
    sys.exit(1)
if not out_dir:
    out_dir = os.path.dirname(rollout_path)
total = success = called_finish = reward_gt0 = 0
rewards = []
with open(rollout_path) as f:
    for line in f:
        try:
            r = json.loads(line)
            total += 1
            if r.get("error") is None: success += 1
            if r.get("called_finish_tool"): called_finish += 1
            rew = r.get("reward", 0)
            if rew > 0: reward_gt0 += 1
            rewards.append(rew)
        except Exception: pass
avg_reward = sum(rewards) / len(rewards) if rewards else 0
summary = {"total": total, "success": success, "called_finish_tool": called_finish, "reward_gt0": reward_gt0, "avg_reward": round(avg_reward, 4)}
out_path = os.path.join(out_dir, "eval_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print("Eval summary written to", out_path)
