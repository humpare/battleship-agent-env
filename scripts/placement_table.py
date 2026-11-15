"""For every legal layout, how many misses does the exact shooter make before sinking it?
The reference uses this to place its own fleet: "put my ships where a perfect counter of
possibilities would look last". Self-play reasoning from the public rules only.

Usage: BATTLESHIP_N=5 uv run python scripts/placement_table.py   -> src/battleship/placement_5.json
"""
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

from battleship import duel
from battleship.duel import N, name, parse
from battleship.duel_reference import ExactPlayer, groups

EX = None


TIE_SEEDS = range(8)  # average over shooters with different tie-break orders: no single habit to exploit


def cost(i):
    global EX
    if EX is None:
        EX = ExactPlayer()
    m = int(EX.all[i])
    cells = [(r, c) for r in range(N) for c in range(N) if m >> (r * N + c) & 1]
    ships = [sorted(g) for g in groups(cells)]
    total_miss = total_shots = 0
    for seed in TIE_SEEDS:
        order = [(r, c) for r in range(N) for c in range(N)]
        random.Random(seed).shuffle(order)
        EX.tie = {x: k for k, x in enumerate(order)}
        side = duel.Side(ships)
        while not side.dead:
            c = parse(EX.choose({"opponent_board": side.public_rows(), "opponent_ships_afloat": side.afloat()}))
            total_miss += side.receive(c)["result"] == "miss"
            total_shots += 1
    n = len(TIE_SEEDS)
    return i, round(total_miss / n, 3), round(total_shots / n, 3), [[name(*x) for x in sorted(g)] for g in ships]


if __name__ == "__main__":
    n = len(ExactPlayer().all)
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        rows = pool.map(cost, range(n), chunksize=20)
    rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    hist = {}
    for r in rows:
        hist[round(r[1])] = hist.get(round(r[1]), 0) + 1
    out = {"board": N, "wrap": bool(__import__("battleship.duel", fromlist=["WRAP"]).WRAP), "fleet": duel.FLEET, "layouts": n, "misses_histogram": dict(sorted(hist.items())),
           "best": [{"misses": r[1], "shots": r[2], "ships": r[3]} for r in rows[:50]]}
    from battleship.duel import WRAP
    path = Path(__file__).resolve().parents[1] / "src" / "battleship" / f"placement_{N}{'w' if WRAP else ''}.json"
    path.write_text(json.dumps(out, indent=1))
    print("layouts", n, "misses histogram", out["misses_histogram"])
    print("hardest:", out["best"][0])
