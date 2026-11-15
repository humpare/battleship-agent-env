"""Play coded players against the frozen boards offline (no Docker), through the same service
the MCP server wraps. Usage: uv run python scripts/offline_eval.py [scenarios/battleship/*.json]"""
import glob
import json
import statistics
import sys

from battleship.duel import DuelService
from battleship.duel_reference import CheckerboardAgent, ExactPlayer, RandomAgent


def play(sc, player, max_shots=None):
    svc = DuelService(sc)
    obs = svc.place_fleet(player.place())["observation"]
    n = 0
    while not obs["game_over"] and (max_shots is None or n < max_shots):
        obs = svc.fire(player.choose(obs), str(n))["observation"]
        n += 1
    return svc.game.result()


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob("scenarios/battleship/bs5-s*.json"))
    scs = [json.load(open(f)) for f in files]
    exact = ExactPlayer()
    rows = {"reference (exact search + placement table)": lambda sc: play(sc, exact),
            "exact search, random placement": None, "checkerboard hunt, random placement": None,
            "random shots, random placement": None, "stop after 4 shots": lambda sc: play(sc, exact, 4)}

    class Mixed:  # exact shooting, random legal placement
        def __init__(self, seed):
            self.p = CheckerboardAgent(seed)
        place = lambda self: self.p.place()  # noqa: E731
        choose = lambda self, obs: exact.choose(obs)  # noqa: E731
    rows["exact search, random placement"] = lambda sc: play(sc, Mixed(sc["seed"]))
    rows["checkerboard hunt, random placement"] = lambda sc: play(sc, CheckerboardAgent(sc["seed"]))
    rows["random shots, random placement"] = lambda sc: play(sc, RandomAgent(sc["seed"]))
    print(f"{len(scs)} boards")
    for label, fn in rows.items():
        res = [fn(sc) for sc in scs]
        print(f"{label:45s} mean score {statistics.mean(r['score'] for r in res):.3f}   wins {sum(r['won'] for r in res)}/{len(res)}")
