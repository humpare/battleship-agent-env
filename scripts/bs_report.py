"""Summarise Battleship runs from Harbor job directories: outcome, uncertainty, and a grade for
every shot against the exact count of remaining fleets.

Usage: uv run python scripts/bs_report.py reference=runs/jobs/X kimi=runs/jobs/Y ... [--json out.json]
For each shot the game log lets us rebuild what the player knew; the exact player's count then
gives every cell's chance of a hit. shot quality = chance at the chosen cell / best chance.
"""
import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from battleship.duel import Duel
from battleship.duel_reference import ExactPlayer, N, read

judge = ExactPlayer()
ROOT = Path(__file__).resolve().parents[1]


def chances(rows):
    left = judge.consistent(rows)
    seen = read(rows)
    shot = set(seen["o"]) | set(seen["x"]) | set(seen["#"])
    return {f"{'ABCDEFGHI'[c]}{r + 1}": float(((left >> np.uint64(r * N + c)) & np.uint64(1)).mean())
            for r in range(N) for c in range(N) if (r, c) not in shot}, len(left)


def grade(scenario: dict, log: Path) -> dict:
    g, out = Duel(scenario), {"shots": [], "fleet": None, "refused": 0}
    for line in log.read_text().splitlines():
        rec = json.loads(line)
        if rec["type"] == "refused":
            out["refused"] += 1
        elif rec["type"] == "place":
            g.place(rec["ships"])
            out["fleet"] = rec["ships"]
        elif rec["type"] == "fire" and not g.over:
            ch, n = chances(g.enemy.public_rows())
            best = max(ch.values())
            out["shots"].append({"cell": rec["target"], "chance": ch[rec["target"]], "best": best, "fleets_left": n,
                                 "result": rec["your_shot"]["result"]})
            g.fire(rec["target"])
    return out


def read_job(job: Path) -> list[dict]:
    rows = []
    for trial in sorted(p for p in job.iterdir() if p.is_dir()):
        if not (trial / "result.json").exists():
            continue
        res = json.loads((trial / "result.json").read_text())
        task = Path(res.get("config", {}).get("task", {}).get("path", res.get("task_name", ""))).name
        reward = ((res.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        v = trial / "verifier" / "result.json"
        detail = json.loads(v.read_text()) if v.exists() else {}
        row = {"task": task, "trial": trial.name, "reward": reward, "won": bool(detail.get("won")),
               "finished": bool(detail.get("finished")),
               "error": (res.get("exception_info") or {}).get("exception_type")}
        log, sc = trial / "artifacts" / "data" / "episode.jsonl", ROOT / "scenarios" / "battleship" / f"{task}.json"
        if log.exists() and sc.exists():
            row.update(grade(json.loads(sc.read_text()), log))
        rows.append(row)
    return rows


def interval(by_board: dict[str, float], n=4000) -> tuple[float, float]:
    """95% bootstrap interval of the mean over boards (boards resampled; runs averaged within a board)."""
    vals, rng = list(by_board.values()), random.Random(0)
    means = sorted(statistics.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["reward"] is not None]
    board = {}
    for r in ok:
        board.setdefault(r["task"], []).append(r["reward"])
    board = {t: statistics.mean(v) for t, v in board.items()}
    shots = [s for r in ok for s in r.get("shots", [])]
    opening = [s for s in shots if s["fleets_left"] > 100]
    late = [s for s in shots if s["fleets_left"] <= 100]
    q = lambda ss: round(statistics.mean(s["chance"] / s["best"] for s in ss), 3) if ss else None  # noqa: E731
    lo, hi = interval(board) if len(board) > 1 else (None, None)
    fleets = [json.dumps(sorted(map(sorted, r["fleet"]))) for r in ok if r.get("fleet")]
    return {"boards": len(board), "runs": len(ok), "errors": len(rows) - len(ok),
            "mean_score": round(statistics.mean(board.values()), 3) if board else None,
            "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None,
            "wins": sum(r["won"] for r in ok), "finished": sum(r["finished"] for r in ok),
            "shot_quality": q(shots), "shot_quality_over_100_fleets_left": q(opening),
            "shot_quality_100_or_fewer_left": q(late),
            "shots_at_under_5pct_cells": round(sum(s["chance"] < 0.05 for s in shots) / len(shots), 3) if shots else None,
            "shots_at_certain_water": round(sum(s["chance"] == 0 for s in shots) / len(shots), 3) if shots else None,
            "first_shot": dict(Counter(r["shots"][0]["cell"] for r in ok if r.get("shots")).most_common(5)),
            "distinct_fleets": len(set(fleets)), "most_common_fleet_runs": Counter(fleets).most_common(1)[0][1] if fleets else 0,
            "fleets_with_a_ship_on_A1": sum('"A1"' in f for f in fleets),
            "refused_requests": sum(r.get("refused", 0) for r in ok), "by_board": board}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    for spec in a.jobs:
        label, _, path = spec.rpartition("=")
        out[label or Path(path).name] = summarise(read_job(Path(path)))
    keys = [k for k in next(iter(out.values())) if k != "by_board"]
    width = max(len(k) for k in keys) + 2
    print(f"{'':{width}}" + "".join(f"{c:>22}" for c in out))
    for k in keys:
        print(f"{k:{width}}" + "".join(f"{json.dumps(s[k]) if isinstance(s[k], (list, dict)) else str(s[k]):>22.22}" for s in out.values()))
    ref = next(iter(out.values()))["by_board"]
    for label, s in list(out.items())[1:]:
        common = [t for t in s["by_board"] if t in ref]
        if common:
            d = {t: ref[t] - s["by_board"][t] for t in common}
            lo, hi = interval(d) if len(d) > 1 else (0, 0)
            print(f"gap {next(iter(out))} - {label}: {statistics.mean(d.values()):.3f} "
                  f"[95% {lo:.3f}, {hi:.3f}] on {len(common)} shared boards; "
                  f"{label} ahead on {sum(v < 0 for v in d.values())} boards")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
