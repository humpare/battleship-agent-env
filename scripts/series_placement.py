"""How much does adaptive placement help over a series against the same opponent?
For each opponent and placement method: the opponent's misses per game (each miss is a turn for us),
for games 1..5 of a series. One-sided on purpose: it isolates placement.

Usage: BATTLESHIP_N=5 uv run python scripts/series_placement.py [series]
"""
import copy
import json
import random
import statistics
import sys
import time
from pathlib import Path

from battleship import duel
from battleship.adaptive import AdaptivePlacer, ships_of
from battleship.duel import N, Side, parse
from battleship.duel_reference import ExactPlayer, _ortho, halo_cells

series_n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
GAMES = 5
base = ExactPlayer(tie_seed=0)
ALL = [(r, c) for r in range(N) for c in range(N)]
mid = N // 2


class HabitShooter:
    """A shooter with a fixed hunting habit (an order of cells, the rest random), who finishes
    wounded ships and skips cells next to sunk ships. Stands in for a model's observed habits."""

    def __init__(self, first: list[tuple], seed: int):
        self.first, self.rng = first, random.Random(seed)

    def play(self, ships):
        side = Side(ships)
        rest = [c for c in ALL if c not in self.first]
        self.rng.shuffle(rest)
        order, log, hits, dead = self.first + rest, [], [], set()
        while not side.dead:
            cand = [y for h in hits for y in _ortho(h) if y not in side.shots and y not in dead]
            if len(hits) >= 2:
                rows, cols = {h[0] for h in hits}, {h[1] for h in hits}
                cand = [y for y in cand if (len(rows) == 1 and y[0] in rows) or (len(cols) == 1 and y[1] in cols)] or cand
            target = self.rng.choice(cand) if cand else next(c for c in order if c not in side.shots and c not in dead)
            res = side.receive(target)["result"]
            log.append({"cell": target, "result": res})
            if res == "hit":
                hits.append(target)
            elif res == "sunk":
                ship = next(s for s in side.ships if target in s)
                dead |= halo_cells(ship)
                hits = [h for h in hits if h not in ship]
        return log


class ExactShooter:
    def __init__(self, seed):
        self.p = copy.copy(base)
        order = list(ALL)
        random.Random(seed).shuffle(order)
        self.p.tie = {x: i for i, x in enumerate(order)}

    def play(self, ships):
        side, log = Side(ships), []
        while not side.dead:
            c = parse(self.p.choose({"opponent_board": side.public_rows(), "opponent_ships_afloat": side.afloat()}))
            log.append({"cell": c, "result": side.receive(c)["result"]})
        return log


OPPONENTS = {
    "centre then diagonals (Opus-like)": lambda s: HabitShooter([(mid, mid), (1, 1), (N - 2, N - 2), (1, N - 2), (N - 2, 1)], s),
    "centre then corners (Kimi/Fable-like)": lambda s: HabitShooter([(mid, mid), (0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1)], s),
    "exact algorithm (no habits)": lambda s: ExactShooter(s),
}
table = Path(duel.__file__).with_name(f"placement_{N}{'w' if duel.WRAP else ''}.json")
TABLE = json.loads(table.read_text())["best"] if table.exists() else None
static = [[parse(c) for c in ship] for ship in TABLE[0]["ships"]] if TABLE else None

print(f"board {N}x{N}{' wrap-around' if duel.WRAP else ''}, {len(base.all):,} fleets, {series_n} series of {GAMES} games")
print(f"{'opponent':40}{'placement':22}" + "".join(f"{'game ' + str(g + 1):>8}" for g in range(GAMES)) + f"{'mean':>8}")
timing = []
for oname, mk in OPPONENTS.items():
    for method in ("random", "static table", "adaptive"):
        if method == "static table" and static is None:
            continue
        per_game = [[] for _ in range(GAMES)]
        for s in range(series_n):
            rng = random.Random(s)
            placer = AdaptivePlacer(base.all, seed=s, table=TABLE)
            shooter = mk(1000 + s)
            history = []
            for g in range(GAMES):
                if method == "random":
                    ships = ships_of(int(base.all[rng.randrange(len(base.all))]))
                elif method == "static table":
                    ships = static
                else:
                    t = time.perf_counter()
                    ships = [[parse(c) for c in ship] for ship in placer.place(history)]
                    timing.append(time.perf_counter() - t)
                log = shooter.play(ships)
                history.append(log)
                per_game[g].append(sum(x["result"] == "miss" for x in log))
        means = [statistics.mean(x) for x in per_game]
        print(f"{oname:40}{method:22}" + "".join(f"{m:>8.2f}" for m in means) + f"{statistics.mean(means):>8.2f}", flush=True)
print(f"adaptive placement time per call: mean {statistics.mean(timing) * 1000:.0f} ms, max {max(timing) * 1000:.0f} ms")
