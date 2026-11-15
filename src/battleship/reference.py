"""Coded players. They see only what the tools return (rules + observation).

DensityPlayer: samples many fleet layouts consistent with everything observed so far and
shoots the unknown cell that holds a ship in the most samples.
CheckerboardPlayer: a decent human-style strategy, as a yardstick: finish wounded ships,
otherwise hunt on a checkerboard, never shoot next to a sunk ship.
"""
from __future__ import annotations

import random

SIZE = 10
COLS = "ABCDEFGHIJ"


def name(r, c):
    return f"{COLS[c]}{r + 1}"


def read_board(obs: dict):
    grid = [row[3:] for row in obs["board"][1:]]
    miss, hit, sunk = set(), set(), set()
    for r in range(SIZE):
        for c in range(SIZE):
            ch = grid[r][c]
            if ch == "o":
                miss.add((r, c))
            elif ch == "x":
                hit.add((r, c))
            elif ch == "#":
                sunk.add((r, c))
    return miss, hit, sunk


def neighbours8(cells):
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < SIZE and 0 <= cc < SIZE:
                    out.add((rr, cc))
    return out


class DensityPlayer:
    def __init__(self, samples: int = 400, seed: int = 0, salvo: int = 1):
        self.samples, self.rng, self.salvo = samples, random.Random(seed), salvo

    def _placements(self, n, free, hit):
        """All straight placements of length n on cells that may hold a ship."""
        out = []
        for r in range(SIZE):
            for c in range(SIZE):
                for dr, dc in ((0, 1), (1, 0)):
                    if n == 1 and dr == 1:
                        continue
                    cells = [(r + dr * i, c + dc * i) for i in range(n)]
                    if all(x in free for x in cells) and not all(x in hit for x in cells):
                        out.append(cells)  # a ship whose cells are all hit would already be announced sunk
        return out

    def _sample(self, lengths, free, hit, cache):
        """One random layout of the remaining ships consistent with the board, or None."""
        placed, blocked, todo_hits = [], set(), set(hit)
        for n in sorted(lengths, reverse=True):
            options = [p for p in cache[n] if not any(x in blocked for x in p)]
            covering = [p for p in options if any(x in todo_hits for x in p)]
            remaining_after = sum(1 for m in lengths if m >= 2) - sum(1 for p in placed if len(p) >= 2) - (1 if n >= 2 else 0)
            if todo_hits and n >= 2 and covering and (remaining_after == 0 or self.rng.random() < 0.75):
                options = covering
            if not options:
                return None
            p = self.rng.choice(options)
            placed.append(p)
            blocked |= neighbours8(p)
            todo_hits -= set(p)
        return None if todo_hits else placed

    def choose(self, obs: dict) -> list[str]:
        miss, hit, sunk = read_board(obs)
        known_empty = miss | (neighbours8(sunk) - sunk)
        free = {(r, c) for r in range(SIZE) for c in range(SIZE)} - known_empty - sunk
        lengths = obs["ships_afloat_by_length"]
        cache = {n: self._placements(n, free, hit) for n in set(lengths)}
        count, got, tries = {}, 0, 0
        while got < self.samples and tries < self.samples * 30:
            tries += 1
            layout = self._sample(lengths, free, hit, cache)
            if layout is None:
                continue
            got += 1
            for p in layout:
                for x in p:
                    if x not in hit:
                        count[x] = count.get(x, 0) + 1
        unknown = sorted(free - hit)
        if not count:  # sampling failed (very constrained board): fall back to cells next to hits
            cand = [x for x in unknown if any(abs(x[0] - h[0]) + abs(x[1] - h[1]) == 1 for h in hit)] or unknown
            return [name(*self.rng.choice(cand))]
        ranked = sorted(unknown, key=lambda x: (-count.get(x, 0), x))
        k = max(1, min(self.salvo, obs["shots_left"], len(ranked)))
        if hit:
            k = 1  # finishing a wounded ship: every answer changes the next best shot
        return [name(*x) for x in ranked[:k]]


class CheckerboardPlayer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, obs: dict) -> list[str]:
        miss, hit, sunk = read_board(obs)
        known_empty = miss | (neighbours8(sunk) - sunk)
        unknown = [(r, c) for r in range(SIZE) for c in range(SIZE)
                   if (r, c) not in known_empty and (r, c) not in sunk and (r, c) not in hit]
        if hit:
            rows, cols = {h[0] for h in hit}, {h[1] for h in hit}
            cand = []
            for r, c in unknown:
                adj = [h for h in hit if abs(h[0] - r) + abs(h[1] - c) == 1]
                if not adj:
                    continue
                if len(hit) >= 2 and len(rows) == 1 and r not in rows:
                    continue
                if len(hit) >= 2 and len(cols) == 1 and c not in cols:
                    continue
                cand.append((r, c))
            if cand:
                return [name(*self.rng.choice(cand))]
        big = max(obs["ships_afloat_by_length"])
        parity = [x for x in unknown if (x[0] + x[1]) % (2 if big >= 2 else 1) == 0]
        return [name(*self.rng.choice(parity or unknown))]


class RandomPlayer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, obs: dict) -> list[str]:
        miss, hit, sunk = read_board(obs)
        unknown = [(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in miss | hit | sunk]
        return [name(*self.rng.choice(unknown))]
