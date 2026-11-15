"""Full five-game series: a coded LEARNING player (stand-in for a model that reads the history)
against the algorithm, old version and new version. Reports the player's wins by game number.

The learning player, like Opus in the recorded series:
  - game 1: a corner-heavy fleet; later games: the fleet least exposed to where the algorithm
    fired its early shots so far, never reusing its own previous layout;
  - shooting: first the cells of the algorithm's previously revealed fleets (most recent first),
    then corners and edges; finishes wounded ships; skips cells next to sunk ships.

Usage: BATTLESHIP_N=5 uv run python scripts/series_duel.py [series]
"""
import copy
import json
import random
import statistics
import sys
from pathlib import Path

from battleship import duel
from battleship.adaptive import AdaptivePlacer, hunt_shots
from battleship.duel import Duel, N, fleet_names, parse
from battleship.duel_reference import ExactPlayer, _ortho, halo_cells, read

series_n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
GAMES = 5
base = ExactPlayer()
ALL = [(r, c) for r in range(N) for c in range(N)]
TABLE = json.loads(Path(duel.__file__).with_name(f"placement_{N}.json").read_text())["best"]
CORNER_FLEET = [["E1", "E2", "E3"], ["A1", "B1"], ["A3", "A4"], ["C5"], ["E5"]]
heat = {c: float(((base.all >> __import__("numpy").uint64(c[0] * N + c[1])) & __import__("numpy").uint64(1)).mean()) for c in ALL}


class Learner:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.placer = AdaptivePlacer(base.all, seed=seed)  # used only as "least exposed to their shots"
        self.algo_shots, self.algo_fleets, self.own_fleets = [], [], []

    def place(self):
        if not self.algo_shots:
            fleet = CORNER_FLEET
        else:
            fleet = self.placer.place(self.algo_shots, self.own_fleets)
        self.own_fleets.append(fleet)
        return fleet

    def start_game(self):
        seen = []
        for fleet in reversed(self.algo_fleets):
            for ship in fleet:
                for cell in ship:
                    c = parse(cell)
                    if c not in seen:
                        seen.append(c)
        rest = sorted((c for c in ALL if c not in seen), key=lambda c: (-heat[c], self.rng.random()))
        self.order = seen + rest

    def choose(self, obs):
        seen = read(obs["opponent_board"])
        sunk, hit = set(seen["#"]), list(seen["x"])
        dead = set(seen["o"]) | (halo_cells(sunk) - sunk)
        shot = dead | sunk | set(hit)
        cand = [y for h in hit for y in _ortho(h) if y not in shot]
        if len(hit) >= 2:
            rows, cols = {h[0] for h in hit}, {h[1] for h in hit}
            cand = [y for y in cand if (len(rows) == 1 and y[0] in rows) or (len(cols) == 1 and y[1] in cols)] or cand
        target = self.rng.choice(cand) if cand else next(c for c in self.order if c not in shot)
        return duel.name(*target)


class Algo:
    """The algorithm as opponent. version 1 = as played against Opus; version 2 = with the fixes."""

    def __init__(self, seed, version):
        self.version, self.seed = version, seed
        self.placer = AdaptivePlacer(base.all, seed=seed, table=TABLE, pick_from=5 if version == 1 else 12)
        self.player_shots, self.player_fleets, self.own_fleets = [], [], []
        self.search = copy.copy(base)

    def new_game(self, g):
        order = list(ALL)
        random.Random(self.seed * 100 + g).shuffle(order)
        self.search.tie = {x: i for i, x in enumerate(order)}
        if self.version == 2:
            self.search.learn_placement(self.player_fleets)
            fleet = self.placer.place(self.player_shots, self.own_fleets)
        else:
            self.search.weights = None
            fleet = self.placer.place(self.player_shots)
        self.own_fleets.append(fleet)
        return fleet

    def choose(self, target):
        return parse(self.search.choose({"opponent_board": target.public_rows(), "opponent_ships_afloat": target.afloat()}))


def ordered(side):
    seen, out = set(), []
    for cell, res in side.shots.items():
        seen.add(cell)
        if res == "hit":
            ship = next(s for s in side.ships if cell in s)
            res = "sunk" if all(x in seen for x in ship) else "hit"
        out.append({"cell": cell, "result": res})
    return out


for version in (1, 2):
    wins = [[] for _ in range(GAMES)]
    margins = [[] for _ in range(GAMES)]
    first_miss = [[] for _ in range(GAMES)]
    for s in range(series_n):
        learner, algo = Learner(s), Algo(1000 + s, version)
        for g in range(GAMES):
            d = Duel({"id": "x", "seed": g, "opponent_fleet": algo.new_game(g), "opponent_shot_seed": g})
            d.bot = algo
            learner.start_game()
            d.place(learner.place())
            while not d.over:
                d.fire(learner.choose(d.observation()))
            r = d.result()
            wins[g].append(r["won"])
            margins[g].append(r["margin"])
            algo_log, player_log = ordered(d.mine), ordered(d.enemy)
            k = 0
            for x in algo_log:
                if x["result"] != "miss":
                    break
                k += 1
            first_miss[g].append(k)
            learner.algo_shots.append(algo_log)
            learner.algo_fleets.append(fleet_names(d.enemy.ships))
            algo.player_shots.append(player_log)
            algo.player_fleets.append(fleet_names(d.mine.ships))
    label = "algorithm v1 (as played against Opus)" if version == 1 else "algorithm v2 (learns placement, burns revealed cells)"
    print(label)
    print("   learning player's win rate by game: " + "  ".join(f"g{g + 1} {statistics.mean(wins[g]):.2f}" for g in range(GAMES))
          + f"   | games 2-5: {statistics.mean(x for g in range(1, GAMES) for x in wins[g]):.2f}")
    print("   player's mean margin by game:        " + "  ".join(f"g{g + 1} {statistics.mean(margins[g]):+.2f}" for g in range(GAMES)))
    print("   algorithm's misses before first hit: " + "  ".join(f"g{g + 1} {statistics.mean(first_miss[g]):.1f}" for g in range(GAMES)), flush=True)
