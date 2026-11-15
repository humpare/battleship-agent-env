"""Sanity check: the reference against itself. Both sides search exactly and both place with the
reference's own placement rule (a fleet from the best-value table). The side that shoots first
should win a little over half the games. Usage: uv run python scripts/mirror_check.py [GAMES]"""
import json
import random
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

from battleship.duel import DuelService
from battleship.duel_reference import ExactPlayer

BEST = json.loads((Path(__file__).resolve().parents[1] / "src/battleship/placement_5_duel.json").read_text())["best"]
P = None


def game(seed):
    global P
    P = P or ExactPlayer()
    rng = random.Random(seed)
    order = [(r, c) for r in range(5) for c in range(5)]
    rng.shuffle(order)
    P.tie = {x: i for i, x in enumerate(order)}  # the first player's own tie-break order
    top = [b["ships"] for b in BEST[:8]]
    sc = {"id": f"mirror{seed}", "opponent": "exact", "opponent_fleet": rng.choice(top), "opponent_shot_seed": seed}
    svc = DuelService(sc)
    obs = svc.place_fleet(rng.choice(top))["observation"]
    n = 0
    while not obs["game_over"]:
        obs = svc.fire(P.choose(obs), str(n))["observation"]
        n += 1
    return svc.game.result()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    with Pool(16) as pool:
        res = pool.map(game, range(n))
    wins = sum(r["won"] for r in res)
    print(f"reference vs reference, {n} games: first player wins {wins} ({100 * wins / n:.1f}%), "
          f"mean score {statistics.mean(r['score'] for r in res):.3f}")
