"""Turn recorded games (Harbor job directories) into one JSON file for the replay page.

For every step it stores both boards, the exact chance of a hit in every cell as the shooter
sees it (the agent's view of the enemy board, and the opponent's view of the agent's board),
and a grade for each agent shot against the best available cell.

Usage: uv run python scripts/make_replays.py OUT.json label=runs/jobs/JOB [label=arena:results/others/DIR ...]
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

from battleship.duel import CELLS, Side, check_fleet, name, parse
from battleship.duel_reference import ExactPlayer, N, read

ROOT = Path(__file__).resolve().parents[1]
judge = ExactPlayer()


def chances(rows):
    left = judge.consistent(rows)
    seen = read(rows)
    shot = set(seen["o"]) | set(seen["x"]) | set(seen["#"])
    heat = [None if (r, c) in shot or len(left) == 0 else round(float(((left >> np.uint64(r * N + c)) & np.uint64(1)).mean()), 4)
            for r in range(N) for c in range(N)]
    return heat, int(len(left))


def fleet_values(fleets):
    """Expected game score of each fleet against the disclosed opponent (as scripts/placement_duel.py)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import placement_duel as pd
    shooter = [pd.shooter_trace(s) for s in range(150)]
    index = {int(m): i for i, m in enumerate(judge.all)}
    out = {}
    for key, ships in fleets.items():
        mask = 0
        for s in ships:
            for cell in s:
                r, c = parse(cell)
                mask |= 1 << (r * N + c)
        _, traces, _ = pd.fleet_trace(index[mask])
        v, w = pd.value(traces, shooter)
        out[key] = {"expected_score": round(v, 3), "win_chance": round(w, 3)}
    return out


def build(label, trial):
    res = json.loads((trial / "result.json").read_text())
    task = Path(res.get("config", {}).get("task", {}).get("path", res.get("task_name", ""))).name
    sc = json.loads((ROOT / "scenarios" / "battleship" / f"{task}.json").read_text())
    log = trial / "artifacts" / "data" / "episode.jsonl"
    if not log.exists():
        return None
    recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    v = trial / "verifier" / "result.json"
    final = json.loads(v.read_text()) if v.exists() else {}
    return frames_from(label, task, trial.name, sc["opponent_fleet"], recs, final)


def build_arena(label, path):
    """A game from the local harness (earlier version of the task). Only the agent's actions and
    the final result were recorded, so the opponent's shots are re-derived by replaying the game;
    the replay is kept only if it reproduces the recorded final result exactly."""
    from battleship.duel import Duel, ExactOpponent
    rec = json.loads(Path(path).read_text())
    want = {k: rec[k] for k in ("score", "won", "own_cells_left", "opponent_shots", "enemy_cells_hit")}
    seed = int(rec["scenario"].split("-")[-1]) if rec["scenario"].split("-")[-1].isdigit() else 0
    for variant, shot_seed in (("board-order", None), ("seeded", seed * 7919 + 13), ("seeded-raw", seed)):
        sc = {"id": rec["scenario"], "opponent": "exact", "opponent_fleet": rec["opponent_fleet"], "opponent_shot_seed": shot_seed or 0}
        g = Duel(sc)
        if shot_seed is None:  # the first version of the opponent broke ties in board order, A1 first
            g.bot.player.tie = {(r, c): r * N + c for r in range(N) for c in range(N)}
        recs = []
        try:
            for a in rec["actions"]:
                if a["type"] == "place":
                    g.place(a["ships"])
                    recs.append({"type": "place", "ships": a["ships"]})
                elif not g.over:
                    out = g.fire(a["target"])
                    recs.append({"type": "fire", "target": a["target"], **out, "score_after": g.result()["score"]})
        except ValueError:
            continue
        got = g.result()
        if all(got[k] == want[k] for k in want):
            return frames_from(label, Path(path).parent.name.rsplit("-", 1)[0], Path(path).stem, rec["opponent_fleet"], recs,
                               {**rec, "winner": rec.get("winner")})
    return None


def frames_from(label, task, trial_name, opponent_fleet, recs, final):
    place = next((r for r in recs if r["type"] == "place"), None)
    if place is None:
        return None
    enemy, mine = Side(check_fleet(opponent_fleet)), Side(check_fleet(place["ships"]))
    frames, score = [], 0.0

    def snap(extra):
        ha, na = chances(enemy.public_rows())
        ho, no = chances(mine.public_rows())
        frames.append({"enemy": [r[2:] for r in enemy.public_rows()[1:]], "mine": [r[2:] for r in mine.own_rows()[1:]],
                       "heat_agent": ha, "fleets_agent": na, "heat_opp": ho, "fleets_opp": no, "score": score, **extra})

    snap({"who": "start", "text": "Fleets placed. The agent shoots first."})
    for rec in recs:
        if rec["type"] != "fire":
            continue
        heat, n = chances(enemy.public_rows())
        best = max(h for h in heat if h is not None)
        cell = parse(rec["target"])
        ch = heat[cell[0] * N + cell[1]]
        best_cells = [name(i // N, i % N) for i, h in enumerate(heat) if h is not None and abs(h - best) < 1e-9]
        would_hit = [b for b in best_cells if parse(b) in enemy.owner]
        r = enemy.receive(cell)
        score = rec.get("score_after", score)
        info = {"who": "agent", "cell": rec["target"], "result": r["result"], "chance": ch, "best": best,
                "best_cells": best_cells, "is_best": abs(ch - best) < 1e-9, "fleets_before": n,
                "best_would_hit": bool(would_hit)}
        if info["is_best"]:
            info["text"] = f"Agent fires {rec['target']}: {r['result']}. A best cell ({100 * ch:.0f}% chance, {n} fleets possible)."
        else:
            alt = ", ".join(best_cells[:3])
            info["text"] = (f"Agent fires {rec['target']}: {r['result']}. It had a {100 * ch:.0f}% chance; the best cell was {alt} "
                            f"at {100 * best:.0f}% ({n} fleets possible)" + (", and it was a ship." if would_hit else "."))
        snap(info)
        for s in rec["opponent_shots"]:
            heat, n = chances(mine.public_rows())
            c2 = parse(s["cell"])
            ch2 = heat[c2[0] * N + c2[1]]
            mine.receive(c2)
            snap({"who": "opponent", "cell": s["cell"], "result": s["result"], "chance": ch2, "fleets_before": n,
                  "text": f"Opponent fires {s['cell']}: {s['result']}. Its most likely cell ({100 * ch2:.0f}% chance, {n} fleets possible)."})
    agent_frames = [f for f in frames if f["who"] == "agent"]
    mistakes = sorted((i for i, f in enumerate(frames) if f["who"] == "agent" and not f["is_best"]),
                      key=lambda i: -(frames[i]["best"] - frames[i]["chance"]) * (2 if frames[i]["result"] == "miss" and frames[i]["best_would_hit"] else 1))
    return {"label": label, "board": task, "trial": trial_name, "score": final.get("score"), "won": final.get("won"),
            "winner": final.get("winner"), "fleet": place["ships"], "enemy_fleet": opponent_fleet,
            "corner_ships": sum(any(c in ("A1", "E1", "A5", "E5") for c in s) for s in place["ships"]),
            "shots": len(agent_frames), "not_best": sum(not f["is_best"] for f in agent_frames),
            "mistakes": mistakes[:4], "frames": frames}


if __name__ == "__main__":
    out, games = Path(sys.argv[1]), []
    for spec in sys.argv[2:]:
        label, _, path = spec.partition("=")
        if path.startswith("arena:"):
            for f in sorted(Path(path[6:]).glob("game-*.json")):
                g = build_arena(label, f)
                print("arena", f, "reproduced" if g else "NOT reproduced, skipped")
                if g:
                    g["earlier"] = True
                    games.append(g)
            continue
        for trial in sorted(p for p in Path(path).iterdir() if p.is_dir() and (p / "result.json").exists()):
            g = build(label, trial)
            if g:
                games.append(g)
    fleets = {json.dumps(sorted(map(sorted, g["fleet"]))): g["fleet"] for g in games}
    vals = fleet_values(fleets)
    for g in games:
        g["fleet_value"] = vals[json.dumps(sorted(map(sorted, g["fleet"])))]
    out.write_text(json.dumps({"n": N, "best_fleet_value": 0.99, "mean_fleet_value": 0.59, "games": games}, separators=(",", ":")))
    for g in games:
        print(g["label"], g["board"], g["score"], "fleet value", g["fleet_value"]["expected_score"], "not best", g["not_best"], "/", g["shots"], "mistake frames", g["mistakes"])
    print("wrote", out, out.stat().st_size, "bytes")
