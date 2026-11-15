"""Adaptive fleet placement for a series of games against the same opponent.

Idea: the ideal placement is the best reply to what the opponent will do. We do not know that,
but after each game we have seen every shot it fired. From those shots we estimate, for every
cell, how early this opponent tends to shoot there while hunting, and blend that with a prior
(where a good searcher looks first). Then we look for the legal fleet this estimated opponent
finds slowest.

Fast by construction (two stages):
  1. exposure score for EVERY legal fleet at once: the summed hunting priority of its cells
     (one vectorised pass per board cell over the full fleet list);
  2. a Monte-Carlo check with a simplified model of the opponent on the best few hundred only.
The answer is near-best, not provably best: stage 2 uses a simplified opponent and few trials.

Uses only public information: the rules and the shots the opponent fired at us in past games.
"""
from __future__ import annotations

import random

import numpy as np

from .duel_reference import N, _ortho, groups, halo_cells, name

ALL_CELLS = [(r, c) for r in range(N) for c in range(N)]


def _parse(cell: str) -> tuple:
    return (int(cell[1:]) - 1, "ABCDEFGHI".index(cell[0]))


def statistics_mean(xs: list) -> float:
    return sum(xs) / len(xs)


def cells_of(mask: int) -> list[tuple]:
    return [(r, c) for r in range(N) for c in range(N) if mask >> (r * N + c) & 1]


def ships_of(mask: int) -> list[list[tuple]]:
    return [sorted(g) for g in groups(cells_of(mask))]


def hunt_shots(game: list[dict]) -> list[tuple]:
    """The opponent's shots that were 'hunting' (fired with no wounded ship on the board).
    game: its shots at us in order, each {"cell": (r, c), "result": "miss"|"hit"|"sunk"}."""
    out, wounded = [], 0
    for s in game:
        if wounded == 0:
            out.append(s["cell"])
        if s["result"] == "hit":
            wounded += 1
        elif s["result"] == "sunk":
            wounded = 0
    return out


class OpponentModel:
    """Per-cell hunting priority = prior + what we observed, with earlier shots counting more."""

    def __init__(self, prior: dict[tuple, float], history: list[list[dict]], weight: float = 3.0,
                 own_past_fleets: list[list[list[tuple]]] | None = None, burn: float = 2.5):
        """own_past_fleets: the fleets WE used in earlier games. They were revealed to the opponent
        when those games ended, so we assume it will shoot there early ("burned" cells), the most
        recent fleet most of all."""
        self.games = len(history)
        z = max(prior.values()) or 1.0
        self.priority = {c: prior[c] / z for c in ALL_CELLS}  # prior scaled to 0..1
        for game in history:
            for rank, cell in enumerate(hunt_shots(game)):
                self.priority[cell] += weight / (1.0 + rank)  # first hunting shot counts most
        past = own_past_fleets or []
        for age, fleet in enumerate(reversed(past)):  # age 0 = the most recent game
            for ship in fleet:
                for cell in ship:
                    self.priority[cell] += burn / (1.0 + age)

    def sample_order(self, rng: random.Random, sharpness: float = 4.0) -> list[tuple]:
        """A plausible hunting order: high-priority cells tend to come first (Gumbel trick)."""
        keyed = [(sharpness * self.priority[c] - np.log(-np.log(rng.random() or 1e-12)), c) for c in ALL_CELLS]
        return [c for _, c in sorted(keyed, reverse=True)]

    def misses(self, ships: list[list[tuple]], rng: random.Random) -> int:
        """Simulate this opponent against one fleet; return how many times it misses
        (each miss is a turn for us). Hunts by sampled priority, then finishes wounded ships,
        and never shoots next to a ship it has sunk."""
        owner = {c: i for i, s in enumerate(ships) for c in s}
        left = [len(s) for s in ships]
        order = self.sample_order(rng)
        shot, dead, hits, misses, afloat = set(), set(), [], 0, len(ships)
        pos = 0
        while afloat:
            target = None
            if hits:
                rows, cols = {h[0] for h in hits}, {h[1] for h in hits}
                cand = [y for h in hits for y in _ortho(h) if y not in shot and y not in dead]
                if len(hits) >= 2:
                    line = [y for y in cand if (len(rows) == 1 and y[0] in rows) or (len(cols) == 1 and y[1] in cols)]
                    cand = line or cand
                if cand:
                    target = rng.choice(cand)
            while target is None:
                c = order[pos]
                pos += 1
                if c not in shot and c not in dead:
                    target = c
            shot.add(target)
            i = owner.get(target)
            if i is None:
                misses += 1
                continue
            left[i] -= 1
            if left[i] == 0:
                afloat -= 1
                dead |= halo_cells(ships[i])
                hits = [h for h in hits if owner[h] != i]
            else:
                hits.append(target)
        return misses


class AdaptivePlacer:
    def __init__(self, layouts: np.ndarray, seed: int = 0, shortlist: int = 200, trials: int = 12,
                 pick_from: int = 5, table: list[dict] | None = None, table_weight: float = 2.0):
        """layouts: every legal fleet as an occupancy bit mask (the same list the exact search uses).
        shortlist/trials set the speed-accuracy trade-off; pick_from > 1 keeps us unpredictable.
        table: optional [{"ships": [[cell names]], "misses": x}] from scripts/placement_table.py,
        the fleets a good general searcher finds slowest. They are always considered, and they
        count for more the fewer games we have seen (with no history the table decides)."""
        self.table = table or []
        self.table_weight = table_weight
        self.layouts = layouts
        self.rng = random.Random(seed)
        self.shortlist, self.trials, self.pick_from = shortlist, trials, pick_from
        # prior: share of legal fleets with a ship in each cell (where a good searcher looks first)
        self.prior = {c: float(((layouts >> np.uint64(c[0] * N + c[1])) & np.uint64(1)).mean()) for c in ALL_CELLS}
        self.last = {}

    def exposure(self, model: OpponentModel) -> np.ndarray:
        """Stage 1: for every fleet, the summed hunting priority of the cells it occupies."""
        total = np.zeros(len(self.layouts), dtype=np.float32)
        for c in ALL_CELLS:
            bit = ((self.layouts >> np.uint64(c[0] * N + c[1])) & np.uint64(1)).astype(np.float32)
            total += np.float32(model.priority[c]) * bit
        return total

    def place(self, history: list[list[dict]], own_past_fleets: list[list[list[str]]] | None = None) -> list[list[str]]:
        """history: the opponent's shots at us in each previous game of the series (may be empty).
        own_past_fleets: the fleets we placed in those games (cell names), already seen by the opponent.
        Fully determined by the seed, the history and the past fleets."""
        past = [[[_parse(c) for c in ship] for ship in fleet] for fleet in (own_past_fleets or [])]
        used = {frozenset(c for ship in fleet for c in ship) for fleet in past}
        model = OpponentModel(self.prior, history, own_past_fleets=past)
        exposure = self.exposure(model)
        k = min(self.shortlist, len(self.layouts))
        best_idx = np.argpartition(exposure, k - 1)[:k]  # least exposed fleets
        candidates = [(ships_of(int(self.layouts[i])), None) for i in best_idx]
        candidates += [([[_parse(c) for c in ship] for ship in t["ships"]], t["misses"]) for t in self.table]
        general = statistics_mean([t["misses"] for t in self.table]) if self.table else 0.0
        trust = self.table_weight / (1.0 + model.games)  # how much the general table still counts
        scored = []
        for j, (ships, table_misses) in enumerate(candidates):
            if frozenset(c for ship in ships for c in ship) in used:
                continue  # never reuse a fleet the opponent has already seen
            mean = sum(model.misses(ships, self.rng) for _ in range(self.trials)) / self.trials
            bonus = trust * ((table_misses if table_misses is not None else general * 0.6) - general)
            scored.append((mean + bonus, mean, j, ships))
        scored.sort(reverse=True)
        choice = scored[self.rng.randrange(min(self.pick_from, len(scored)))]
        self.last = {"games_seen": model.games, "expected_opponent_misses": round(choice[1], 2),
                     "best_expected": round(scored[0][1], 2), "candidates": len(candidates)}
        return [[name(*c) for c in s] for s in choice[3]]
