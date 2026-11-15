"""Battleship (classic Russian Battleship), one-sided: the agent shoots at a fixed hidden fleet.

Standard rules: 10x10 board, fleet of one 4-cell, two 3-cell, three 2-cell and four 1-cell
ships, straight, never touching (not even diagonally). Each shot is answered miss / hit / sunk.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from pathlib import Path

SIZE = 10
COLS = "ABCDEFGHIJ"
FLEET = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]
SHIP_CELLS = sum(FLEET)

RULES = """\
BATTLESHIP. A fleet is hidden on a 10x10 board. Sink it with as few shots as you can.

BOARD
- Columns A to J (left to right), rows 1 to 10 (top to bottom). A cell is written like "C7".

FLEET (always exactly this)
- One ship of 4 cells, two of 3 cells, three of 2 cells, four of 1 cell: ten ships, 20 cells.
- Every ship is a straight horizontal or vertical line.
- Ships never touch each other, not even diagonally: every cell next to a ship (including
  diagonal neighbours) is empty water.
- The fleet is fixed before the game starts and does not move.

SHOOTING
- fire(targets, request_id): targets is a list of 1 to {max_salvo} cells, shot in the order given.
  Each shot is answered "miss", "hit" (the ship has more cells left) or "sunk" (that shot
  finished the ship; its length is reported).
- You may not shoot a cell twice. If any target in a call is invalid, the whole call is refused
  and nothing is shot.
- You have {budget} shots in total. The game ends when the whole fleet is sunk or the shots run out.
- request_id is any string unique to the call; repeating it returns the first answer again
  without shooting twice.

WHAT YOU SEE
- observe(): the board ("." unknown, "o" miss, "x" hit on a ship not yet sunk, "#" sunk ship),
  shots used and left, and which ship sizes are still afloat. It does not use a shot.

SCORING
- Your result is the share of the 20 ship cells you have hit when the game ends: 1.0 means the
  whole fleet was sunk within the shot budget, 0 means nothing was hit. You are not shown a score.
"""


def cell_name(r: int, c: int) -> str:
    return f"{COLS[c]}{r + 1}"


def parse_cell(s) -> tuple[int, int] | None:
    if not isinstance(s, str):
        return None
    s = s.strip().upper()
    if len(s) < 2 or s[0] not in COLS or not s[1:].isdigit():
        return None
    r = int(s[1:]) - 1
    return (r, COLS.index(s[0])) if 0 <= r < SIZE else None


def halo(cells) -> set:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < SIZE and 0 <= cc < SIZE:
                    out.add((rr, cc))
    return out


def random_fleet(seed: int) -> list[list[list[int]]]:
    rng = random.Random(seed)
    while True:
        ships, blocked, ok = [], set(), True
        for n in FLEET:
            for _ in range(500):
                horiz = rng.random() < 0.5
                r = rng.randrange(SIZE - (0 if horiz else n - 1))
                c = rng.randrange(SIZE - (n - 1 if horiz else 0))
                cells = [(r, c + i) if horiz else (r + i, c) for i in range(n)]
                if not any(x in blocked for x in cells):
                    ships.append(cells)
                    blocked |= halo(cells)
                    break
            else:
                ok = False
                break
        if ok:
            return [[list(x) for x in s] for s in ships]


def make_scenario(sid: str, seed: int, budget: int, max_salvo: int = 5) -> dict:
    return {"id": sid, "seed": seed, "budget": budget, "max_salvo": max_salvo, "fleet": random_fleet(seed)}


def scenario_hash(sc: dict) -> str:
    return hashlib.sha256(json.dumps(sc, sort_keys=True).encode()).hexdigest()[:16]


class Game:
    def __init__(self, scenario: dict):
        self.id = scenario["id"]
        self.budget = int(scenario["budget"])
        self.max_salvo = int(scenario["max_salvo"])
        self.ships = [[tuple(c) for c in s] for s in scenario["fleet"]]
        assert sorted(len(s) for s in self.ships) == sorted(FLEET)
        self.owner = {c: i for i, s in enumerate(self.ships) for c in s}
        assert len(self.owner) == SHIP_CELLS
        for i, s in enumerate(self.ships):  # no touching
            assert not any(self.owner.get(c, i) != i for c in halo(s))
        self.shots: dict[tuple, str] = {}
        self.sunk: set[int] = set()
        self.turn = 0

    @property
    def over(self) -> bool:
        return len(self.shots) >= self.budget or len(self.sunk) == len(self.ships)

    def check(self, targets) -> list[tuple]:
        if not isinstance(targets, list) or not 1 <= len(targets) <= self.max_salvo:
            raise ValueError(f"targets must be a list of 1 to {self.max_salvo} cells")
        cells = []
        for t in targets:
            c = parse_cell(t)
            if c is None:
                raise ValueError(f"{t!r} is not a cell (use A1 to J10)")
            if c in self.shots or c in cells:
                raise ValueError(f"{cell_name(*c)} has already been shot")
            cells.append(c)
        if len(self.shots) + len(cells) > self.budget:
            raise ValueError(f"only {self.budget - len(self.shots)} shots left")
        return cells

    def fire(self, targets) -> list[dict]:
        cells = self.check(targets)
        out = []
        for c in cells:
            if len(self.sunk) == len(self.ships):
                break  # fleet already gone: remaining targets are not spent
            i = self.owner.get(c)
            if i is None:
                self.shots[c] = "miss"
                out.append({"cell": cell_name(*c), "result": "miss"})
                continue
            self.shots[c] = "hit"
            if all(x in self.shots for x in self.ships[i]):
                self.sunk.add(i)
                out.append({"cell": cell_name(*c), "result": "sunk", "ship_length": len(self.ships[i])})
            else:
                out.append({"cell": cell_name(*c), "result": "hit"})
        self.turn += 1
        return out

    def observation(self) -> dict:
        rows = []
        sunk_cells = {c for i in self.sunk for c in self.ships[i]}
        for r in range(SIZE):
            line = ""
            for c in range(SIZE):
                s = self.shots.get((r, c))
                line += "." if s is None else "o" if s == "miss" else "#" if (r, c) in sunk_cells else "x"
            rows.append(f"{r + 1:>2} {line}")
        afloat = sorted((len(s) for i, s in enumerate(self.ships) if i not in self.sunk), reverse=True)
        return {"game_id": self.id, "turn": self.turn, "board": ["   " + COLS] + rows,
                "legend": ". unknown, o miss, x hit (ship not sunk yet), # sunk ship",
                "shots_used": len(self.shots), "shots_left": self.budget - len(self.shots),
                "ships_afloat_by_length": afloat, "game_over": self.over}

    def result(self) -> dict:
        hits = sum(1 for v in self.shots.values() if v == "hit")
        return {"scenario": self.id, "score": round(hits / SHIP_CELLS, 6), "ship_cells_hit": hits,
                "ships_sunk": len(self.sunk), "fleet_sunk": len(self.sunk) == len(self.ships),
                "shots_used": len(self.shots), "budget": self.budget, "turns": self.turn}


def rules_for(game: Game) -> dict:
    return {"rules": RULES.replace("{budget}", str(game.budget)).replace("{max_salvo}", str(game.max_salvo)),
            "board_size": SIZE, "fleet_lengths": FLEET, "shot_budget": game.budget, "max_targets_per_call": game.max_salvo}


class Service:
    """The three tools. The private log of accepted salvos is what the verifier replays."""

    def __init__(self, scenario: dict, log_path=None):
        self.game = Game(scenario)
        self.lock = threading.Lock()
        self.answers: dict[str, dict] = {}
        self.log_path = Path(log_path) if log_path else None
        self.salvos: list[list[str]] = []
        self._write({"type": "header", "scenario": self.game.id, "scenario_hash": scenario_hash(scenario)})

    def _write(self, rec):
        if self.log_path:
            rec["wall_time"] = round(time.time(), 3)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    def get_rules(self) -> dict:
        return rules_for(self.game)

    def observe(self) -> dict:
        with self.lock:
            return self.game.observation()

    def fire(self, targets, request_id) -> dict:
        with self.lock:
            g = self.game
            if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
                return {"status": "invalid_request", "message": "request_id must be a non-empty string",
                        "observation": g.observation()}
            if request_id in self.answers:
                return {**self.answers[request_id], "duplicate_request": True}
            if isinstance(targets, str):
                try:
                    targets = json.loads(targets) if targets.strip().startswith("[") else [targets]
                except ValueError:
                    targets = None
            if g.over:
                return {"status": "game_over", "message": "the game has ended", "observation": g.observation()}
            try:
                shots = g.fire(targets)
            except ValueError as e:
                self._write({"type": "refused", "message": str(e)})
                return {"status": "rejected", "message": str(e), "shots": [], "observation": g.observation()}
            self.salvos.append([s["cell"] for s in shots] if shots else [])
            self._write({"type": "salvo", "turn": g.turn, "targets": [parse_and_name(t) for t in targets],
                         "shots": shots, "score_after": g.result()["score"]})
            ans = {"status": "accepted", "shots": shots, "observation": g.observation()}
            self.answers[request_id] = ans
            return ans


def parse_and_name(t) -> str:
    c = parse_cell(t)
    return cell_name(*c) if c else str(t)


def load_salvos(log_path) -> tuple[dict | None, list[list[str]]]:
    header, salvos = None, []
    for line in Path(log_path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("type") == "header" and header is None:
                header = rec
            elif rec.get("type") == "salvo":
                salvos.append(rec["targets"])
    return header, salvos


def replay(scenario: dict, salvos: list[list[str]]) -> dict:
    g = Game(scenario)
    for targets in salvos:
        if g.over:
            break
        g.fire(targets)
    return g.result()
