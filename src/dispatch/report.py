"""Summarise Harbor job directories into one table.

Usage: python -m dispatch.report runs/jobs/JOB [runs/jobs/JOB ...] [--json out.json]
Each column is one job (label = job directory name, or NAME=PATH to rename).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_job(job: Path) -> list[dict]:
    rows = []
    for trial in sorted(p for p in job.iterdir() if p.is_dir()):
        res_file = trial / "result.json"
        if not res_file.exists():
            continue
        res = json.loads(res_file.read_text())
        cfg = res.get("config", {}).get("agent", {})
        detail = {}
        vfile = trial / "verifier" / "result.json"
        if vfile.exists():
            detail = json.loads(vfile.read_text())
        rewards = (res.get("verifier_result") or {}).get("rewards") or {}
        agent_result = res.get("agent_result") or {}
        rows.append({
            "task": Path(res.get("config", {}).get("task", {}).get("path", res.get("task_name", ""))).name,
            "trial": trial.name, "agent": cfg.get("name"), "model": cfg.get("model_name"),
            "reward": rewards.get("reward"),
            "completed": detail.get("completed"), "trains": detail.get("trains"),
            "total_lateness": detail.get("total_lateness"),
            "deadlocked": len((detail.get("diagnostics") or {}).get("deadlocked_trains", [])),
            "accepted_steps": detail.get("accepted_steps"),
            "exception": (res.get("exception_info") or {}).get("exception_type"),
            "input_tokens": agent_result.get("n_input_tokens"), "output_tokens": agent_result.get("n_output_tokens"),
            "cost_usd": agent_result.get("cost_usd"),
            "started_at": res.get("started_at"), "finished_at": res.get("finished_at"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cols, data = [], {}
    for spec in a.jobs:
        label, _, path = spec.rpartition("=")
        path = Path(path)
        label = label or path.name
        cols.append(label)
        for r in read_job(path):
            data.setdefault(r["task"], {}).setdefault(label, []).append(r)
    width = max((len(t) for t in data), default=10) + 2

    def cell(rows: list[dict]) -> str:
        vals = [r["reward"] for r in rows if r["reward"] is not None]
        if not vals:
            return "error" if rows else "-"
        if len(vals) == 1:
            r = rows[0]
            extra = f" ({r['completed']}/{r['trains']})" if r["completed"] is not None else ""
            return f"{vals[0]:.3f}{extra}"
        return f"{statistics.mean(vals):.3f} [{min(vals):.2f}-{max(vals):.2f}] n={len(vals)}"

    print(f"{'task':{width}}" + "".join(f"{c:>26}" for c in cols))
    means = {c: [] for c in cols}
    for task in sorted(data):
        line = f"{task:{width}}"
        for c in cols:
            rows = data[task].get(c, [])
            line += f"{cell(rows):>26}"
            means[c] += [r["reward"] for r in rows if r["reward"] is not None]
        print(line)
    print(f"{'MEAN':{width}}" + "".join(f"{(f'{statistics.mean(v):.3f}' if v else '-'):>26}" for v in means.values()))
    if a.json:
        Path(a.json).write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
