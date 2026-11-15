"""Verifier: recompute the score from the action log. Trusts nothing the agent wrote."""
import argparse
import json
import sys
from pathlib import Path

from dispatch.service import load_batches
from dispatch.sim import replay, scenario_hash

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def finish(score: float, detail: dict) -> None:
    (out / "reward.txt").write_text(f"{score:.6f}")
    (out / "result.json").write_text(json.dumps(detail, indent=2))
    print(json.dumps({k: v for k, v in detail.items() if k != "per_train"}))
    sys.exit(0)


scenario = json.loads(Path(a.scenario).read_text())
log = Path(a.log)
if not log.exists():
    finish(0.0, {"score": 0.0, "error": "no action log collected from the simulator"})
try:
    header, batches = load_batches(log)
except Exception as e:  # unreadable log: no credit
    finish(0.0, {"score": 0.0, "error": f"action log unreadable: {e}"})
if header is None or header.get("scenario_hash") != scenario_hash(scenario):
    finish(0.0, {"score": 0.0, "error": "action log does not belong to this scenario"})
try:
    result = replay(scenario, batches)
except Exception as e:  # a log the simulator itself would never have accepted
    finish(0.0, {"score": 0.0, "error": f"action log does not replay: {e}"})
result["accepted_steps"] = len(batches)
finish(result["score"], result)
