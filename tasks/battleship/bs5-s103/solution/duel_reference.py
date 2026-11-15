"""Coded players for the Battleship duel. They use only the rules and the observation.

ExactPlayer keeps the complete list of every legal opponent fleet layout (5,772 on the 5x5
task board, about 12.9 million on 7x7), deletes the ones that contradict what has been seen, and picks the shot with the best
expected chain of hits over the layouts that remain (a hit keeps the turn, so chains matter).
"""
from __future__ import annotations

import random

import numpy as np

import os as _os

N = int(_os.environ.get("BATTLESHIP_N", "5"))  # board size; the tasks use 5
COLS = "ABCDEFGHI"[:N]
WRAP = _os.environ.get("BATTLESHIP_WRAP", "0") == "1"
FLEET = [int(x) for x in _os.environ.get("BATTLESHIP_FLEET", "3,2,2,1,1").split(",")]


def name(r, c):
    return f"{COLS[c]}{r + 1}"


def bit(r, c):
    return np.uint64(1) << np.uint64(r * N + c)


def _placements(n):
    masks, halos, seen = [], [], set()
    for r in range(N):
        for c in range(N):
            for dr, dc in ((0, 1), (1, 0)):
                if n == 1 and dr:
                    continue
                cells = [(r + dr * i, c + dc * i) for i in range(n)]
                if WRAP:
                    cells = [(x % N, y % N) for x, y in cells]
                elif not all(0 <= x < N and 0 <= y < N for x, y in cells):
                    continue
                m = h = 0
                for x, y in cells:
                    m |= 1 << (x * N + y)
                    for a in (-1, 0, 1):
                        for b in (-1, 0, 1):
                            if WRAP:
                                h |= 1 << (((x + a) % N) * N + (y + b) % N)
                            elif 0 <= x + a < N and 0 <= y + b < N:
                                h |= 1 << ((x + a) * N + y + b)
                if m in seen or bin(m).count("1") != n:
                    continue
                seen.add(m)
                masks.append(m)
                halos.append(h)
    return np.array(masks, dtype=np.uint64), np.array(halos, dtype=np.uint64)


def _ortho(x):
    out = []
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        y = (x[0] + d[0], x[1] + d[1])
        if WRAP:
            out.append((y[0] % N, y[1] % N))
        elif 0 <= y[0] < N and 0 <= y[1] < N:
            out.append(y)
    return out


def all_layouts() -> np.ndarray:
    """Every legal fleet layout as a 49-bit occupancy mask."""
    mask = np.zeros(1, dtype=np.uint64)
    blocked = np.zeros(1, dtype=np.uint64)
    last = np.full(1, -1, dtype=np.int64)
    prev_len = None
    for n in sorted(FLEET, reverse=True):
        pm, ph = _placements(n)
        idx = np.arange(len(pm))
        out_m, out_b, out_l = [], [], []
        for s in range(0, len(mask), 200_000):
            m, b, l = mask[s:s + 200_000], blocked[s:s + 200_000], last[s:s + 200_000]
            ok = (b[:, None] & pm[None, :]) == 0
            if n == prev_len:  # identical ships: only count each unordered pair once
                ok &= idx[None, :] > l[:, None]
            i, j = np.nonzero(ok)
            out_m.append(m[i] | pm[j])
            out_b.append(b[i] | ph[j])
            out_l.append(j.astype(np.int64))
        mask, blocked, last = np.concatenate(out_m), np.concatenate(out_b), np.concatenate(out_l)
        prev_len = n
    return mask


def read(rows):
    grid = [r[2:] for r in rows[1:]]
    out = {"o": [], "x": [], "#": []}
    for r in range(N):
        for c in range(N):
            if grid[r][c] in out:
                out[grid[r][c]].append((r, c))
    return out


def groups(cells):
    cells, out = set(cells), []
    while cells:
        stack, g = [cells.pop()], []
        while stack:
            x = stack.pop()
            g.append(x)
            for y in _ortho(x):
                if y in cells:
                    cells.remove(y)
                    stack.append(y)
        out.append(g)
    return out


def mask_of(cells):
    m = np.uint64(0)
    for r, c in cells:
        m |= bit(r, c)
    return m


def halo_cells(cells):
    if WRAP:
        return {((r + a) % N, (c + b) % N) for r, c in cells for a in (-1, 0, 1) for b in (-1, 0, 1)}
    return {(r + a, c + b) for r, c in cells for a in (-1, 0, 1) for b in (-1, 0, 1)
            if 0 <= r + a < N and 0 <= c + b < N}


class ExactPlayer:
    def __init__(self, depth: int = 2, top: int = 6, seed: int = 0, sample: int = 0, tie_seed: int | None = None):
        """tie_seed: when several cells are exactly equally good, which one to take. None keeps
        board order (A1 first), which is predictable; an integer gives a fixed random order, so the
        player has no favourite cells but is still fully reproducible."""
        self.sample = sample
        self.tie = {(r, c): r * N + c for r in range(N) for c in range(N)}
        if tie_seed is not None:
            order = list(self.tie)
            random.Random(tie_seed).shuffle(order)
            self.tie = {x: i for i, x in enumerate(order)}
        self.all = all_layouts()
        self.weights = None  # optional prior weight per fleet (learned from the opponent's past fleets)
        self.depth, self.top = depth, top
        self.rng = random.Random(seed)

    def place(self) -> list[list[str]]:
        """Place the fleet with the best expected score for the whole game against a searcher
        that counts possibilities (table built by scripts/placement_duel.py: it keeps my ships out
        of the cells such a searcher shoots first, so its early turns are misses while my own
        search finishes). Falls back to the older table (most misses to sink everything,
        scripts/placement_table.py), then to a random legal fleet."""
        import json
        from pathlib import Path
        tag = f"{N}{'w' if WRAP else ''}"
        for table in (Path(__file__).with_name(f"placement_{tag}_duel.json"), Path(__file__).with_name(f"placement_{tag}.json")):
            if table.exists():
                t = json.loads(table.read_text())
                if sorted(len(x) for x in t["best"][0]["ships"]) == sorted(FLEET):
                    return t["best"][0]["ships"]
        m = int(self.all[self.rng.randrange(len(self.all))])
        cells = [(r, c) for r in range(N) for c in range(N) if m >> (r * N + c) & 1]
        return [[name(*x) for x in sorted(g)] for g in groups(cells)]

    def consistent(self, rows) -> np.ndarray:
        seen = read(rows)
        a = self.all
        keep = (a & mask_of(seen["o"])) == 0
        for ship in groups(seen["#"]):  # a sunk ship: exactly these cells, water all around
            keep &= (a & mask_of(halo_cells(ship))) == mask_of(ship)
        if seen["x"]:
            keep &= (a & mask_of(seen["x"])) == mask_of(seen["x"])
            for g in groups(seen["x"]):  # a wounded ship must continue into a cell not yet shot
                nb = {y for x in g for y in _ortho(x)} - set(g)
                keep &= (a & mask_of(nb)) != 0
        self._keep = keep
        return a[keep]

    def learn_placement(self, past_fleets: list[list[list[str]]], strength: float = 1.0) -> None:
        """Weight every possible enemy fleet by how well it matches where this opponent has put
        ships before. Each cell gets an estimated occupancy chance (past frequency, smoothed
        toward the no-information chance); a fleet's weight is the product of the odds of its
        cells. With no past fleets every fleet weighs the same (the plain exact search)."""
        if not past_fleets:
            self.weights = None
            return
        base = {(r, c): float(((self.all >> np.uint64(r * N + c)) & np.uint64(1)).mean())
                for r in range(N) for c in range(N)}
        n = len(past_fleets)
        seen = {x: 0.0 for x in base}
        for k, fleet in enumerate(past_fleets):  # later games count a little more
            w = 1.0 + 0.5 * k / max(1, n - 1) if n > 1 else 1.0
            for ship in fleet:
                for cell in ship:
                    seen[(int(cell[1:]) - 1, COLS.index(cell[0]))] += w
        total = sum(1.0 + 0.5 * k / max(1, n - 1) if n > 1 else 1.0 for k in range(n))
        prior_games = 2.0  # how many games of evidence the no-information chance is worth
        logw = np.zeros(len(self.all), dtype=np.float64)
        for x, b in base.items():
            p = (seen[x] + prior_games * b) / (total + prior_games)
            p = min(max(p, 0.02), 0.98)
            shift = strength * (np.log(p / (1 - p)) - np.log(min(max(b, 0.02), 0.98) / (1 - min(max(b, 0.02), 0.98))))
            logw += shift * ((self.all >> np.uint64(x[0] * N + x[1])) & np.uint64(1)).astype(np.float64)
        logw -= logw.max()
        self.weights = np.exp(logw)

    def _value(self, a, w, cells, depth):
        """Best expected number of hits in a row from this position, and the shot that gives it.
        a: fleets still possible; w: their weights (None = all equal)."""
        if len(a) == 0 or not cells:
            return 0.0, None
        total = float(len(a)) if w is None else float(w.sum())
        if total <= 0:
            return 0.0, None
        cnt = {}
        for x in cells:
            bit = ((a >> np.uint64(x[0] * N + x[1])) & np.uint64(1))
            cnt[x] = float(bit.sum()) if w is None else float((w * bit).sum())
        ranked = sorted(cells, key=lambda x: (-round(cnt[x], 9), self.tie[x]))
        best, best_cell = -1.0, None
        for x in ranked[:self.top if depth else 1]:
            p = cnt[x] / total
            v = p
            if depth and p > 0:
                sel = ((a >> np.uint64(x[0] * N + x[1])) & np.uint64(1)) == 1
                v += p * self._value(a[sel], None if w is None else w[sel], [y for y in cells if y != x], depth - 1)[0]
            if v > best + 1e-12:
                best, best_cell = v, x
        return best, best_cell

    def choose(self, obs: dict) -> str:
        rows = obs["opponent_board"]
        a = self.consistent(rows)
        seen = read(rows)
        shot = set(seen["o"]) | set(seen["x"]) | set(seen["#"])
        cells = [(r, c) for r in range(N) for c in range(N) if (r, c) not in shot]
        if self.sample and len(a) > self.sample:  # optional speed-up; off by default: every count is exact
            a = a[np.random.default_rng(0).choice(len(a), self.sample, replace=False)]
        self.remaining = len(a)
        w = None if self.weights is None else self.weights[self._keep]
        if self.sample and w is not None and len(w) != len(a):
            w = None  # sampling and learned weights are not combined
        return name(*self._value(a, w, cells, self.depth)[1])


class CheckerboardAgent:
    """Yardstick: the same strategy the opponent uses, as the player."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def place(self):
        from .duel import fleet_names, random_fleet
        return fleet_names(random_fleet(self.rng))

    def choose(self, obs):
        seen = read(obs["opponent_board"])
        sunk, hit = set(seen["#"]), set(seen["x"])
        dead = set(seen["o"]) | (halo_cells(sunk) - sunk)
        unknown = sorted((r, c) for r in range(N) for c in range(N) if (r, c) not in dead | sunk | hit)
        if hit:
            rows, cols = {h[0] for h in hit}, {h[1] for h in hit}
            cand = [x for x in unknown if any(y in hit for y in _ortho(x))
                    and not (len(hit) >= 2 and len(rows) == 1 and x[0] not in rows)
                    and not (len(hit) >= 2 and len(cols) == 1 and x[1] not in cols)]
            if cand:
                return name(*self.rng.choice(cand))
        par = [x for x in unknown if (x[0] + x[1]) % 2 == 0] if max(obs["opponent_ships_afloat"]) >= 2 else unknown
        return name(*self.rng.choice(par or unknown))


class RandomAgent(CheckerboardAgent):
    def choose(self, obs):
        seen = read(obs["opponent_board"])
        shot = set(seen["o"]) | set(seen["x"]) | set(seen["#"])
        return name(*self.rng.choice([(r, c) for r in range(N) for c in range(N) if (r, c) not in shot]))
