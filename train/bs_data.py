"""Training examples for the small model: decisions of the exact-elimination player.

Each game is played against a uniformly drawn enemy fleet that is NOT one of the evaluation
boards. To cover positions a weaker player reaches, a share of the shots are random, but the
label is always what the exact player would do there (ties broken at random).
Usage: train/.venv/bin/python train/bs_data.py --games 4000 --out train/data/bs5.jsonl
"""
import argparse
import json
import random

from bs_common import PLACE, eval_scenarios, messages, shot_prompt

from battleship.duel import Duel, fleet_names, uniform_fleet
from battleship.duel_reference import ExactPlayer, N

ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=4000)
ap.add_argument("--explore", type=float, default=0.25, help="share of shots taken at random (label stays the expert's)")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--depth", type=int, default=2, help="expert look-ahead (0 = plain most-likely cell; use on big boards)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

rng = random.Random(a.seed)
held_out = {json.dumps(sc["opponent_fleet"]) for group in eval_scenarios().values() for sc in group}
expert = ExactPlayer(depth=a.depth)
import pathlib  # noqa: E402
table = pathlib.Path(__file__).resolve().parents[1] / "src" / "battleship" / f"placement_{N}_duel.json"
best = json.loads(table.read_text())["best"] if table.exists() else None
from battleship.duel import FLEET  # noqa: E402
if best and sorted(len(x) for x in best[0]["ships"]) != sorted(FLEET):
    best = None  # the table is for another fleet
cells_all = [(r, c) for r in range(N) for c in range(N)]
rows, shots = [], 0
for g in range(a.games):
    while True:
        fleet = fleet_names(uniform_fleet(rng))
        if json.dumps(fleet) not in held_out:
            break
    order = cells_all[:]
    rng.shuffle(order)
    expert.tie = {x: i for i, x in enumerate(order)}
    sc = {"id": f"train{g}", "opponent": "checkerboard",  # the opponent does not affect the labels
           "opponent_fleet": fleet, "opponent_shot_seed": rng.randrange(10**9)}
    game = Duel(sc)
    mine = rng.choice(best[:8])["ships"] if best else fleet_names(uniform_fleet(rng))
    mine = [list(s) for s in mine]
    rng.shuffle(mine)
    rows.append({"kind": "place", "messages": messages(PLACE, json.dumps(mine, separators=(",", ":")))})
    game.place(mine)
    while not game.over:
        obs = game.observation()
        target = expert.choose(obs)
        rows.append({"kind": "shot", "messages": messages(shot_prompt(obs), target)})
        shots += 1
        if rng.random() < a.explore:
            free = [f"{'ABCDEFGHI'[c]}{r + 1}" for r, c in cells_all if obs["opponent_board"][r + 1][2 + c] == "."]
            target = rng.choice(free)
        game.fire(target)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"{a.games} games, {shots} shot decisions, {len(rows)} examples -> {a.out}")
