"""Choose the reference's fleet by the expected SCORE OF THE WHOLE GAME, from public rules only.

The earlier table (placement_table.py) ranked fleets by how many misses the opponent needs to
sink all of them. But the game is a race: what matters is how much of my fleet is still unhit
when my own search finishes. So for every legal fleet we replay the disclosed opponent against
it (several tie-break orders, since the real order is unknown) and record how many of my cells
are hit after each of its turns; and we replay my own exact search against many random enemy
fleets to learn how many turns I hand over (one per miss). Combining the two gives the expected
score of each fleet.

Usage: uv run python scripts/placement_duel.py   -> src/battleship/placement_5_duel.json
"""
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

from battleship import duel
from battleship.duel import CELLS, N, name, parse, uniform_fleet
from battleship.duel_reference import ExactPlayer, groups

EX = None
TIE_SEEDS = range(12)
SHOOTER_GAMES = 400


def player():
    global EX
    if EX is None:
        EX = ExactPlayer()
    return EX


def shooter_trace(seed):
    """My search against one random enemy fleet: my hits made before each of my misses, and the total."""
    ex = player()
    ex.tie = {(r, c): r * N + c for r in range(N) for c in range(N)}
    side = duel.Side(uniform_fleet(random.Random(10_000 + seed)))
    before, hits = [], 0
    while not side.dead:
        c = parse(ex.choose({"opponent_board": side.public_rows(), "opponent_ships_afloat": side.afloat()}))
        if side.receive(c)["result"] == "miss":
            before.append(hits)
        else:
            hits += 1
    return before


def fleet_trace(i):
    """The opponent against fleet i: my cells hit after each of its turns, per tie-break order."""
    ex = player()
    m = int(ex.all[i])
    cells = [(r, c) for r in range(N) for c in range(N) if m >> (r * N + c) & 1]
    ships = [sorted(g) for g in groups(cells)]
    traces = []
    for seed in TIE_SEEDS:
        order = [(r, c) for r in range(N) for c in range(N)]
        random.Random(seed).shuffle(order)
        ex.tie = {x: k for k, x in enumerate(order)}
        side, after, hits = duel.Side(ships), [], 0
        while not side.dead:
            c = parse(ex.choose({"opponent_board": side.public_rows(), "opponent_ships_afloat": side.afloat()}))
            if side.receive(c)["result"] == "miss":
                after.append(hits)
            else:
                hits += 1
        after.append(CELLS)  # the turn in which it sinks my last ship
        traces.append(after)
    return i, traces, [[name(*x) for x in g] for g in sorted(ships, key=lambda g: (-len(g), g))]


def value(traces, shooter):
    total = wins = 0
    for after in traces:
        for before in shooter:
            turns = len(before)  # the opponent gets one turn per miss of mine
            if turns >= len(after):  # it sinks my fleet in its turn number len(after)
                total += 0.5 * before[len(after) - 1] / CELLS
            else:
                hit = after[turns - 1] if turns else 0
                total += 0.5 + 0.5 * (CELLS - hit) / CELLS
                wins += 1
    n = len(traces) * len(shooter)
    return total / n, wins / n


if __name__ == "__main__":
    n = len(ExactPlayer().all)
    with Pool(min(16, max(1, os.cpu_count() - 4))) as pool:
        shooter = pool.map(shooter_trace, range(SHOOTER_GAMES), chunksize=5)
        rows = pool.map(fleet_trace, range(n), chunksize=10)
    scored = sorted(((*value(t, shooter), ships) for _, t, ships in rows), key=lambda r: -r[0])
    out = {"board": N, "fleet": duel.FLEET, "layouts": n, "tie_orders": len(TIE_SEEDS), "shooter_games": SHOOTER_GAMES,
           "my_mean_misses": round(sum(len(b) for b in shooter) / len(shooter), 2),
           "mean_expected_score_all_fleets": round(sum(r[0] for r in scored) / n, 4),
           "best": [{"expected_score": round(r[0], 4), "win_chance": round(r[1], 4), "ships": r[2]} for r in scored[:50]],
           "worst": [{"expected_score": round(r[0], 4), "win_chance": round(r[1], 4), "ships": r[2]} for r in scored[-5:]]}
    path = Path(__file__).resolve().parents[1] / "src" / "battleship" / f"placement_{N}_duel.json"
    path.write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k not in ("best", "worst")})
    for r in out["best"][:5] + out["worst"][-2:]:
        print(r)
