"""Offline match-ups for one board configuration (set BATTLESHIP_N / BATTLESHIP_WRAP / BATTLESHIP_FLEET).
Players place a random legal fleet, so the numbers isolate shooting skill. Usage: python scripts/sim_matrix.py [games]"""
import random
import statistics
import sys

from battleship import duel
from battleship.duel import Duel, N, WRAP, parse
from battleship.duel_reference import CheckerboardAgent, ExactPlayer, groups

games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
base = ExactPlayer()


def layout(rng):
    m = int(base.all[rng.randrange(len(base.all))])
    cells = [(r, c) for r in range(N) for c in range(N) if m >> (r * N + c) & 1]
    return [sorted(g) for g in groups(cells)]


class ExactAgent:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def place(self):
        return duel.fleet_names(layout(self.rng))

    def choose(self, obs):
        return base.choose(obs)


class CheckerAgent(CheckerboardAgent):
    def place(self):
        return duel.fleet_names(layout(self.rng))


class ExactBot:
    def choose(self, target):
        return parse(base.choose({"opponent_board": target.public_rows(), "opponent_ships_afloat": target.afloat()}))


def play(seed, agent, opponent):
    rng = random.Random(seed)
    g = Duel({"id": "m", "seed": 0, "opponent_fleet": duel.fleet_names(layout(rng)), "opponent_shot_seed": seed})
    if opponent == "exact":
        g.bot = ExactBot()
    g.place(agent.place())
    while not g.over:
        g.fire(agent.choose(g.observation()))
    return g.result()


print(f"board {N}x{N} {'wrap-around' if WRAP else 'with edges'}, fleet {duel.FLEET}, {len(base.all):,} layouts, {games} games per cell")
for opp in ("checkerboard", "exact"):
    for name, mk in (("exact algorithm", ExactAgent), ("checkerboard", CheckerAgent)):
        R = [play(s, mk(s + 10**6), opp) for s in range(1, games + 1)]
        print(f"  {name:16} vs {opp:12} opponent: win rate {statistics.mean(r['won'] for r in R):.3f}  "
              f"mean score {statistics.mean(r['score'] for r in R):.3f}  shots {statistics.mean(r['your_shots'] for r in R):.1f}", flush=True)
rng = random.Random(5)
miss = []
for _ in range(400):
    side = duel.Side(layout(rng))
    k = 0
    while not side.dead:
        c = parse(base.choose({"opponent_board": side.public_rows(), "opponent_ships_afloat": side.afloat()}))
        k += side.receive(c)["result"] == "miss"
    miss.append(k)
miss.sort()
print(f"  exact shooter's misses against 400 random layouts: min {miss[0]}  mean {statistics.mean(miss):.2f}  max {miss[-1]}")
