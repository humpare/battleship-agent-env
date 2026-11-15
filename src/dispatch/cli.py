"""Command line for the offline side of the benchmark (no Docker, no model).

  python -m dispatch.cli validate                         check every scenario: feasible, deterministic, baselines sane
  python -m dispatch.cli run SCENARIO [--policy P] [--max-steps N] [--out DIR]
  python -m dispatch.cli compare [SCENARIO ...]           reference vs yardsticks, one row per scenario
  python -m dispatch.cli replay SCENARIO LOG              re-score a saved action log (same code as the verifier)
  python -m dispatch.cli branch SCENARIO LOG --at K [--alt JSON]
        Restore the state before accepted step K of a saved run, then compare
        (a) the commands the run actually issued there and (b) an alternative
        (default: what the coded reference would have issued in that exact state),
        both continued by the coded reference under identical external events.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .baselines import make
from .reference import ReferenceDispatcher
from .rules import rules
from .runner import run_policy
from .service import load_batches
from .sim import DispatchSim, replay

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ["reference", "serial", "greedy", "nothing"]


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def all_scenarios() -> list[str]:
    return sorted(str(p) for p in (ROOT / "scenarios").glob("*/*.json"))


def brief(r: dict) -> dict:
    return {k: r[k] for k in ("scenario", "score", "completed", "trains", "unfinished", "total_lateness",
                              "final_tick", "decisions")} | {"deadlocked": r["diagnostics"]["deadlocked_trains"]}


def cmd_validate(_a) -> int:
    bad = 0
    for path in all_scenarios():
        sc = load(path)
        net = rules(DispatchSim(sc))["network"]
        ref, log = run_policy(sc)
        again = replay(sc, log)
        serial, _ = run_policy(sc, make("serial", net))
        nothing = replay(sc, [])
        checks = {
            "reference completes every train": ref["unfinished"] == 0,
            "reference has no deadlock": not ref["diagnostics"]["deadlocked_trains"],
            "replay reproduces the score": again["per_train"] == ref["per_train"],
            "one-at-a-time witness completes": serial["unfinished"] == 0,
            "do-nothing scores zero": nothing["score"] == 0.0,
            "reference beats one-at-a-time": ref["score"] >= serial["score"],
        }
        ok = all(checks.values())
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {sc['id']:22} reference={ref['score']:.3f} serial={serial['score']:.3f} "
              f"decisions={ref['decisions']}" + ("" if ok else f"  failed: {[k for k, v in checks.items() if not v]}"))
    print(f"{len(all_scenarios()) - bad}/{len(all_scenarios())} scenarios valid")
    return 1 if bad else 0


def cmd_run(a) -> int:
    sc = load(a.scenario)
    net = rules(DispatchSim(sc))["network"]
    res, log = run_policy(sc, make(a.policy, net), a.max_steps)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(res, indent=1))
        with (out / "episode.jsonl").open("w") as f:
            f.write(json.dumps({"type": "header", "scenario": sc["id"], "scenario_hash": res["scenario_hash"]}) + "\n")
            for i, c in enumerate(log):
                f.write(json.dumps({"type": "step", "decision_id": i + 1, "commands": c}) + "\n")
    print(json.dumps(brief(res) | {"policy": a.policy, "policy_seconds": res["policy_seconds"]}))
    return 0


def cmd_compare(a) -> int:
    paths = a.scenarios or all_scenarios()
    print(f"{'scenario':24}" + "".join(f"{p:>11}" for p in POLICIES) + f"{'ref late':>10}{'ref steps':>10}")
    for path in paths:
        sc = load(path)
        net = rules(DispatchSim(sc))["network"]
        row = {p: run_policy(sc, make(p, net))[0] for p in POLICIES}
        r = row["reference"]
        print(f"{sc['id']:24}" + "".join(f"{row[p]['score']:>11.3f}" for p in POLICIES)
              + f"{r['total_lateness']:>10}{r['decisions']:>10}")
    return 0


def cmd_replay(a) -> int:
    sc = load(a.scenario)
    header, batches = load_batches(a.log)
    res = replay(sc, batches)
    print(json.dumps(brief(res) | {"accepted_steps": len(batches)}))
    if a.verbose:
        print(json.dumps(res["per_train"], indent=1))
    return 0


def cmd_branch(a) -> int:
    sc = load(a.scenario)
    _, batches = load_batches(a.log)
    k = a.at
    if not 0 <= k < len(batches):
        print(f"--at must be between 0 and {len(batches) - 1}", file=sys.stderr)
        return 2

    def restore() -> DispatchSim:
        sim = DispatchSim(sc)
        for c in batches[:k]:
            sim.apply(c)
            sim.advance()
        return sim

    def finish(sim: DispatchSim, decision: list[dict]) -> dict:
        ref = ReferenceDispatcher(rules(sim)["network"])
        sim.apply(decision)
        sim.advance()
        while not sim.over:
            sim.apply(ref.decide(sim.observation()))
            sim.advance()
        return sim.result()

    base = restore()
    obs = base.observation()
    alt = json.loads(a.alt) if a.alt else ReferenceDispatcher(rules(base)["network"]).decide(obs)
    actual = finish(restore(), batches[k])
    other = finish(restore(), alt)
    whole = replay(sc, batches)
    out = {
        "scenario": sc["id"], "decision_index": k, "tick": obs["time"],
        "actual_commands": batches[k], "alternative_commands": alt,
        "continuation": "coded reference from the next decision on, same external events",
        "run_as_played": brief(whole),
        "actual_then_reference": brief(actual),
        "alternative_then_reference": brief(other),
        "difference": {"score": round(other["score"] - actual["score"], 6),
                       "completed": other["completed"] - actual["completed"],
                       "total_lateness": other["total_lateness"] - actual["total_lateness"]},
    }
    print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="dispatch.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    r = sub.add_parser("run")
    r.add_argument("scenario")
    r.add_argument("--policy", default="reference", choices=POLICIES)
    r.add_argument("--max-steps", type=int, default=None)
    r.add_argument("--out", default=None)
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("scenarios", nargs="*")
    c.set_defaults(fn=cmd_compare)
    p = sub.add_parser("replay")
    p.add_argument("scenario")
    p.add_argument("log")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_replay)
    b = sub.add_parser("branch")
    b.add_argument("scenario")
    b.add_argument("log")
    b.add_argument("--at", type=int, required=True, help="index of the accepted step to branch at (0 = first)")
    b.add_argument("--alt", default=None, help="alternative commands as JSON (default: the reference's choice)")
    b.set_defaults(fn=cmd_branch)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
