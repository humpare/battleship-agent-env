"""Recompute the result by replaying the game server's private action log."""
import json
from pathlib import Path

from battleship.duel import load_actions, replay, scenario_hash

out = Path("/logs/verifier")
sc = json.loads(Path("/tests/scenario.json").read_text())
log = Path("/data/episode.jsonl")
res = {"score": 0.0, "error": "no action log collected"}
if log.exists():
    try:
        header, actions = load_actions(log)
        if header and header.get("scenario_hash") == scenario_hash(sc):
            res = replay(sc, actions)
        else:
            res = {"score": 0.0, "error": "log does not belong to this scenario"}
    except Exception as e:
        res = {"score": 0.0, "error": f"log does not replay: {e}"}
(out / "reward.txt").write_text(f"{res['score']:.6f}")
(out / "result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
