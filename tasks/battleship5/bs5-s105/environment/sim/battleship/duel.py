"""Battleship duel: the two-player game on a small board (5x5 by default) against a fixed coded opponent.

Rules: each side has a fleet of ship lengths 3, 2, 2, 1, 1, straight, never touching (not even
diagonally). Players alternate; a hit or a sinking earns another shot, a miss passes the turn.
The first to sink the whole enemy fleet wins. The agent places its own fleet, then shoots first.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from pathlib import Path

import os as _os

N = int(_os.environ.get("BATTLESHIP_N", "5"))  # board size; the tasks use 5
COLS = "ABCDEFGHI"[:N]
WRAP = _os.environ.get("BATTLESHIP_WRAP", "0") == "1"  # wrap-around board (left joins right, top joins bottom)
FLEET = [int(x) for x in _os.environ.get("BATTLESHIP_FLEET", "3,2,2,1,1").split(",")]
CELLS = sum(FLEET)

RULES = """\
BATTLESHIP (two players, {N}x{N} board). You and an opponent each have a hidden fleet. Sink the
opponent's fleet before it sinks yours.

BOARD
- Columns A to {LAST} (left to right), rows 1 to {N} (top to bottom). A cell is written like "C4".

FLEETS (both sides, always exactly this)
- One ship of 3 cells, two ships of 2 cells, two ships of 1 cell: five ships, 9 cells.
- Every ship is a straight horizontal or vertical line.
- Ships never touch each other, not even diagonally: every cell next to a ship (including
  diagonal neighbours) is empty water.
- Fleets do not move once the game starts.

SETUP
- place_fleet(ships): place your own fleet before the first shot. ships is a list of five ships,
  each a list of its cells, e.g. {EXAMPLE}.
  An illegal layout is refused and you may try again. Once accepted it cannot be changed.

PLAY
- You shoot first. fire(target, request_id) shoots ONE cell of the opponent's board.
  The answer is "miss", "hit" (that ship has more cells left) or "sunk" (ship finished; its
  length is reported).
- A hit or a sinking gives you another shot. A miss passes the turn: the opponent then shoots
  at your fleet, and keeps shooting while it hits, until it misses. You see every one of its
  shots in the answer. Then it is your turn again.
- You may not shoot a cell twice; an invalid target is refused and nothing happens.
- The game ends as soon as one fleet is completely sunk.
- request_id is any string unique to the call; repeating it returns the first answer again
  without shooting twice.

WHAT YOU SEE
- observe(): the opponent's board as you know it ("." unknown, "o" miss, "x" hit on a ship not
  yet sunk, "#" sunk ship), your own board ("S" your ship, "X" your ship cell that was hit,
  "o" opponent miss), the ship lengths still afloat on each side, and whose turn it is.

SCORING
- If you win: 0.5 plus 0.5 times the share of your own 9 ship cells still unhit.
- If you lose, or stop before the game ends: 0.5 times the share of the opponent's 9 ship cells
  you hit.
- You are not shown a score.
"""

OPPONENT_RULES = {
    "exact": """
THE OPPONENT
- Its fleet was drawn uniformly at random from all legal fleets (every legal fleet equally likely).
- It is a program and it plays the same way every time. Before each shot it lists every legal
  fleet that agrees with the results of all its shots so far, counts for every cell how many of
  those fleets have a ship there, and shoots the cell that gives the longest expected run of hits
  over its next three shots (nearly always simply the most likely cell). Exact ties are broken in
  an order that is fixed for this game and has no pattern.
- It knows nothing about your board except the results of its own shots.
""",
    "checkerboard": """
THE OPPONENT
- Its fleet was drawn at random from the legal fleets.
- It is a program: it finishes a ship it has hit, never shoots next to a ship it has sunk, and
  otherwise shoots a random cell of one colour of a checkerboard.
""",
}


EXAMPLE_FLEET = [["A1", "A2", "A3"], ["C1", "D1"], ["E3", "E4"], ["A5"], ["C5"]]


WRAP_RULES = """\
WRAP-AROUND BOARD
- This board has no edges: it wraps around. Column A is next to column {LAST}, and row 1 is next
  to row {N}. So "{LAST}3" and "A3" are neighbours, and so are "C1" and "C{N}".
- A ship may run across an edge, e.g. a 3-cell ship on {LAST}2, A2, B2.
- The no-touching rule also applies across edges, including diagonally.

"""


def rules_text(opponent: str | None = None) -> str:
    """The rules given to every player. opponent: also describe how that coded opponent plays."""
    text = RULES + OPPONENT_RULES.get(opponent, "")
    if WRAP:
        text = text.replace("FLEETS (both sides", WRAP_RULES + "FLEETS (both sides")
    return (text.replace("{N}", str(N)).replace("{LAST}", COLS[-1])
            .replace("{EXAMPLE}", json.dumps(EXAMPLE_FLEET, separators=(",", ":"))))


def name(r, c):
    return f"{COLS[c]}{r + 1}"


def parse(s):
    if not isinstance(s, str):
        return None
    s = s.strip().upper()
    if len(s) != 2 or s[0] not in COLS or not s[1].isdigit():
        return None
    r = int(s[1]) - 1
    return (r, COLS.index(s[0])) if 0 <= r < N else None


def halo(cells):
    if WRAP:
        return {((r + a) % N, (c + b) % N) for r, c in cells for a in (-1, 0, 1) for b in (-1, 0, 1)}
    return {(r + a, c + b) for r, c in cells for a in (-1, 0, 1) for b in (-1, 0, 1)
            if 0 <= r + a < N and 0 <= c + b < N}


def adjacent(a, b) -> bool:
    """Orthogonal neighbours (across the seam too on a wrap-around board)."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    if WRAP:
        dr, dc = min(dr, N - dr), min(dc, N - dc)
    return dr + dc == 1


def _run(values: list[int]) -> bool:
    """Do these distinct coordinates form one unbroken run (cyclic on a wrap-around board)?"""
    vs = sorted(values)
    if vs == list(range(vs[0], vs[0] + len(vs))):
        return True
    return WRAP and any({(start + i) % N for i in range(len(vs))} == set(vs) for start in range(N))


def check_fleet(ships) -> list[list[tuple]]:
    """Validate a fleet given as lists of cell names. Returns ships as cell tuples or raises ValueError."""
    if not isinstance(ships, list) or len(ships) != len(FLEET):
        raise ValueError(f"a fleet is a list of {len(FLEET)} ships")
    out, used = [], {}
    for i, ship in enumerate(ships):
        if not isinstance(ship, list) or not ship:
            raise ValueError(f"ship {i + 1} must be a non-empty list of cells")
        cells = [parse(x) for x in ship]
        if any(c is None for c in cells):
            raise ValueError(f"ship {i + 1} has a cell that is not on the board (use A1 to {COLS[-1]}{N})")
        if len(set(cells)) != len(cells):
            raise ValueError(f"ship {i + 1} repeats a cell")
        rows, cols = sorted({c[0] for c in cells}), sorted({c[1] for c in cells})
        line = (len(rows) == 1 and len(cols) == len(cells) and _run(cols)) or \
               (len(cols) == 1 and len(rows) == len(cells) and _run(rows))
        if not line:
            raise ValueError(f"ship {i + 1} is not a straight unbroken line")
        out.append(sorted(cells))
    if sorted(len(s) for s in out) != sorted(FLEET):
        raise ValueError(f"ship lengths must be exactly {sorted(FLEET, reverse=True)}")
    for i, s in enumerate(out):
        for c in halo(s):
            if used.get(c, i) != i:
                raise ValueError(f"ship {i + 1} touches or overlaps another ship")
        for c in s:
            used[c] = i
    return out


def random_fleet(rng: random.Random) -> list[list[tuple]]:
    while True:
        ships, blocked = [], set()
        for n in FLEET:
            for _ in range(300):
                horiz = rng.random() < 0.5
                if WRAP:
                    r, c = rng.randrange(N), rng.randrange(N)
                    cells = [(r, (c + i) % N) if horiz else ((r + i) % N, c) for i in range(n)]
                else:
                    r = rng.randrange(N - (0 if horiz else n - 1))
                    c = rng.randrange(N - (n - 1 if horiz else 0))
                    cells = [(r, c + i) if horiz else (r + i, c) for i in range(n)]
                if not any(x in blocked for x in cells):
                    ships.append(cells)
                    blocked |= halo(cells)
                    break
            else:
                break
        if len(ships) == len(FLEET):
            return ships


def fleet_names(ships):
    return [[name(*c) for c in s] for s in ships]


class Side:
    """One fleet and the shots received."""

    def __init__(self, ships):
        self.ships = [list(map(tuple, s)) for s in ships]
        self.owner = {c: i for i, s in enumerate(self.ships) for c in s}
        self.shots: dict[tuple, str] = {}
        self.sunk: set[int] = set()

    def receive(self, cell) -> dict:
        i = self.owner.get(cell)
        if i is None:
            self.shots[cell] = "miss"
            return {"cell": name(*cell), "result": "miss"}
        self.shots[cell] = "hit"
        if all(x in self.shots for x in self.ships[i]):
            self.sunk.add(i)
            return {"cell": name(*cell), "result": "sunk", "ship_length": len(self.ships[i])}
        return {"cell": name(*cell), "result": "hit"}

    @property
    def dead(self):
        return len(self.sunk) == len(self.ships)

    @property
    def hits(self):
        return sum(v == "hit" for v in self.shots.values())

    def afloat(self):
        return sorted((len(s) for i, s in enumerate(self.ships) if i not in self.sunk), reverse=True)

    def public_rows(self):
        """What the attacker knows."""
        sunk = {c for i in self.sunk for c in self.ships[i]}
        rows = []
        for r in range(N):
            line = ""
            for c in range(N):
                s = self.shots.get((r, c))
                line += "." if s is None else "o" if s == "miss" else "#" if (r, c) in sunk else "x"
            rows.append(f"{r + 1} {line}")
        return ["  " + COLS] + rows

    def own_rows(self):
        rows = []
        for r in range(N):
            line = ""
            for c in range(N):
                s = self.shots.get((r, c))
                ship = (r, c) in self.owner
                line += "X" if s == "hit" else "o" if s == "miss" else "S" if ship else "."
            rows.append(f"{r + 1} {line}")
        return ["  " + COLS] + rows


class Checkerboard:
    """The fixed opponent: finish wounded ships, never shoot next to a sunk ship, otherwise hunt on
    a checkerboard. Deterministic for a given seed and sequence of answers."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def choose(self, target: Side) -> tuple:
        sunk = {c for i in target.sunk for c in target.ships[i]}
        hit = {c for c, v in target.shots.items() if v == "hit"} - sunk
        dead = {c for c, v in target.shots.items() if v == "miss"} | (halo(sunk) - sunk)
        unknown = sorted((r, c) for r in range(N) for c in range(N)
                         if (r, c) not in target.shots and (r, c) not in dead)
        if hit:
            rows, cols = {h[0] for h in hit}, {h[1] for h in hit}
            cand = []
            for x in unknown:
                if not any(adjacent(x, h) for h in hit):
                    continue
                if len(hit) >= 2 and len(rows) == 1 and x[0] not in rows:
                    continue
                if len(hit) >= 2 and len(cols) == 1 and x[1] not in cols:
                    continue
                cand.append(x)
            if cand:
                return self.rng.choice(cand)
        big = max(target.afloat())
        par = [x for x in unknown if (x[0] + x[1]) % 2 == 0] if big >= 2 else unknown
        return self.rng.choice(par or unknown)


class ExactOpponent:
    """The exact-elimination algorithm as the opponent. It is given only what any attacker
    sees (the public view of your board), and it is fully deterministic."""

    _shared = None

    def __init__(self, tie_seed: int = 0):
        import copy

        from .duel_reference import ExactPlayer
        if ExactOpponent._shared is None:
            ExactOpponent._shared = ExactPlayer()
        # Equally good cells are taken in a fixed random order (per scenario), so the opponent
        # has no favourite cells such as A1, yet every run of the scenario is identical.
        self.player = copy.copy(ExactOpponent._shared)
        order = [(r, c) for r in range(N) for c in range(N)]
        random.Random(tie_seed).shuffle(order)
        self.player.tie = {x: i for i, x in enumerate(order)}

    def choose(self, target: Side) -> tuple:
        return parse(self.player.choose({"opponent_board": target.public_rows(),
                                         "opponent_ships_afloat": target.afloat()}))


def choose_opponent_fleet(scenario: dict, history: list) -> list[list[str]]:
    from pathlib import Path

    from .adaptive import AdaptivePlacer
    from .duel_reference import ExactPlayer
    if ExactOpponent._shared is None:
        ExactOpponent._shared = ExactPlayer()
    table = Path(__file__).with_name(f"placement_{N}{'w' if WRAP else ''}.json")
    best = json.loads(table.read_text())["best"] if table.exists() else None
    placer = AdaptivePlacer(ExactOpponent._shared.all, seed=scenario["opponent_shot_seed"], table=best)
    return placer.place(history)


def scenario_hash(sc: dict) -> str:
    return hashlib.sha256(json.dumps(sc, sort_keys=True).encode()).hexdigest()[:16]


def uniform_fleet(rng: random.Random) -> list[list[tuple]]:
    """A fleet drawn uniformly from the complete list of legal fleets (random_fleet, which places
    ship after ship, does not give every legal fleet the same chance)."""
    from .duel_reference import all_layouts, groups
    global _LAYOUTS
    if _LAYOUTS is None:
        _LAYOUTS = all_layouts()
    m = int(_LAYOUTS[rng.randrange(len(_LAYOUTS))])
    cells = [(r, c) for r in range(N) for c in range(N) if m >> (r * N + c) & 1]
    return sorted((sorted(g) for g in groups(cells)), key=lambda g: (-len(g), g))


_LAYOUTS = None


def make_scenario(sid: str, seed: int, opponent: str = "checkerboard", opponent_placement: str = "dealt",
                  uniform: bool = False, disclose_opponent: bool = False) -> dict:
    """uniform: draw the opponent's fleet uniformly from all legal fleets (the frozen evaluation set).
    disclose_opponent: get_rules also tells the player how the opponent places and shoots."""
    rng = random.Random(seed)
    fleet = uniform_fleet(rng) if uniform else random_fleet(rng)
    sc = {"id": sid, "seed": seed, "board": N, "fleet": FLEET, "opponent": opponent,
          "opponent_placement": opponent_placement,
          "opponent_fleet": fleet_names(fleet),
          "opponent_shot_seed": seed * 7919 + 13}
    if disclose_opponent:
        sc["disclose_opponent"] = True
    return sc


class Duel:
    def __init__(self, scenario: dict):
        self.id = scenario["id"]
        if scenario.get("opponent_placement") == "chosen":
            # In the spirit of the game: the opponent picks its own fleet (the kind a good searcher
            # finds slowest), deterministically from the scenario seed. history = its opponent's
            # shots in earlier games of a series, if any.
            self.enemy = Side(check_fleet(choose_opponent_fleet(scenario, scenario.get("history", []))))
        else:
            self.enemy = Side(check_fleet(scenario["opponent_fleet"]))
        if scenario.get("opponent", "checkerboard") == "exact":
            self.bot = ExactOpponent(scenario["opponent_shot_seed"])
        else:
            self.bot = Checkerboard(scenario["opponent_shot_seed"])
        self.mine: Side | None = None
        self.turn = 0
        self.winner: str | None = None

    @property
    def over(self):
        return self.winner is not None

    def place(self, ships):
        if self.mine is not None:
            raise ValueError("your fleet is already placed and cannot be changed")
        self.mine = Side(check_fleet(ships))

    def fire(self, target) -> dict:
        if self.mine is None:
            raise ValueError("place your fleet first with place_fleet")
        if self.over:
            raise ValueError("the game is over")
        cell = parse(target)
        if cell is None:
            raise ValueError(f"{target!r} is not a cell (use A1 to {COLS[-1]}{N})")
        if cell in self.enemy.shots:
            raise ValueError(f"{name(*cell)} has already been shot")
        self.turn += 1
        mine = self.enemy.receive(cell)
        theirs = []
        if self.enemy.dead:
            self.winner = "you"
        elif mine["result"] == "miss":
            while True:
                shot = self.mine.receive(self.bot.choose(self.mine))
                theirs.append(shot)
                if self.mine.dead:
                    self.winner = "opponent"
                    break
                if shot["result"] == "miss":
                    break
        return {"your_shot": mine, "opponent_shots": theirs}

    def observation(self) -> dict:
        o = {"game_id": self.id, "fleet_placed": self.mine is not None, "your_shots_fired": self.turn,
             "opponent_board": self.enemy.public_rows(),
             "opponent_ships_afloat": self.enemy.afloat(),
             "legend": "opponent board: . unknown, o miss, x hit (not sunk), # sunk | "
                       "your board: S ship, X your ship cell hit, o opponent miss",
             "game_over": self.over, "winner": self.winner,
             "next": "game over" if self.over else "your shot" if self.mine is not None else "place your fleet"}
        if self.mine is not None:
            o["your_board"] = self.mine.own_rows()
            o["your_ships_afloat"] = self.mine.afloat()
        return o

    def result(self) -> dict:
        hit_enemy = self.enemy.hits
        own_left = CELLS - (self.mine.hits if self.mine else 0)
        if self.winner == "you":
            score = 0.5 + 0.5 * own_left / CELLS
        else:
            score = 0.5 * hit_enemy / CELLS
        # margin: +1 = enemy fleet sunk with your own untouched, -1 = your fleet sunk with the enemy's untouched
        margin = (own_left - (CELLS - hit_enemy)) / CELLS
        return {"scenario": self.id, "score": round(score, 6), "margin": round(margin, 4),
                "enemy_cells_left": CELLS - hit_enemy, "winner": self.winner, "won": self.winner == "you",
                "finished": self.over, "enemy_cells_hit": hit_enemy, "own_cells_left": own_left,
                "your_shots": self.turn, "opponent_shots": len(self.mine.shots) if self.mine else 0}


class DuelService:
    def __init__(self, scenario: dict, log_path=None):
        self.game = Duel(scenario)
        self.disclosed = scenario.get("opponent", "checkerboard") if scenario.get("disclose_opponent") else None
        self.lock = threading.Lock()
        self.answers: dict[str, dict] = {}
        self.log_path = Path(log_path) if log_path else None
        self.actions: list[dict] = []
        self._write({"type": "header", "scenario": self.game.id, "scenario_hash": scenario_hash(scenario)})

    def _write(self, rec):
        if self.log_path:
            rec["wall_time"] = round(time.time(), 3)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    def get_rules(self):
        return {"rules": rules_text(self.disclosed), "board_size": N, "fleet_lengths": FLEET}

    def observe(self):
        with self.lock:
            return self.game.observation()

    def place_fleet(self, ships):
        with self.lock:
            if isinstance(ships, str):
                try:
                    ships = json.loads(ships)
                except ValueError:
                    ships = None
            try:
                self.game.place(ships)
            except ValueError as e:
                self._write({"type": "refused", "what": "place", "message": str(e)})
                return {"status": "rejected", "message": str(e), "observation": self.game.observation()}
            placed = fleet_names(self.game.mine.ships)
            self.actions.append({"type": "place", "ships": placed})
            self._write({"type": "place", "ships": placed})
            return {"status": "accepted", "observation": self.game.observation()}

    def fire(self, target, request_id):
        with self.lock:
            g = self.game
            if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
                return {"status": "invalid_request", "message": "request_id must be a non-empty string",
                        "observation": g.observation()}
            if request_id in self.answers:
                return {**self.answers[request_id], "duplicate_request": True}
            try:
                res = g.fire(target)
            except ValueError as e:
                self._write({"type": "refused", "what": "fire", "message": str(e)})
                return {"status": "rejected", "message": str(e), "observation": g.observation()}
            cell = res["your_shot"]["cell"]
            self.actions.append({"type": "fire", "target": cell})
            self._write({"type": "fire", "target": cell, **res, "score_after": g.result()["score"]})
            ans = {"status": "accepted", **res, "observation": g.observation()}
            self.answers[request_id] = ans
            return ans


def load_actions(log_path):
    header, actions = None, []
    for line in Path(log_path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("type") == "header" and header is None:
                header = rec
            elif rec.get("type") == "place":
                actions.append({"type": "place", "ships": rec["ships"]})
            elif rec.get("type") == "fire":
                actions.append({"type": "fire", "target": rec["target"]})
    return header, actions


def replay(scenario: dict, actions: list[dict]) -> dict:
    g = Duel(scenario)
    for a in actions:
        if g.over:
            break
        if a["type"] == "place":
            g.place(a["ships"])
        else:
            g.fire(a["target"])
    return g.result()
