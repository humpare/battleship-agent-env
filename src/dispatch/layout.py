"""Named railway layout -> Flatland grid.

A layout is a set of named *sections* (runs of plain track) joined at named
*junctions* (one switch cell each). The public representation given to
controllers is this named graph; the Flatland grid is derived from it and is
never shown to a controller.

Directions follow Flatland: 0=N, 1=E, 2=S, 3=W. Cells are (row, col).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap

Cell = tuple[int, int]
DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


def side_between(a: Cell, b: Cell) -> int:
    """Direction of travel from cell a to the adjacent cell b."""
    d = (b[0] - a[0], b[1] - a[1])
    for k, v in DELTA.items():
        if v == d:
            return k
    raise ValueError(f"cells {a} and {b} are not adjacent")


def rasterize(corners: list[Cell]) -> list[Cell]:
    """Expand an axis-aligned polyline given by its corner cells."""
    cells = [tuple(corners[0])]
    for nxt in corners[1:]:
        cur = cells[-1]
        nxt = tuple(nxt)
        if cur[0] != nxt[0] and cur[1] != nxt[1]:
            raise ValueError(f"polyline segment {cur}->{nxt} is not axis-aligned")
        dr = (nxt[0] > cur[0]) - (nxt[0] < cur[0])
        dc = (nxt[1] > cur[1]) - (nxt[1] < cur[1])
        while cur != nxt:
            cur = (cur[0] + dr, cur[1] + dc)
            cells.append(cur)
    return cells


@dataclass
class Section:
    name: str
    kind: str  # "single" | "loop" | "origin" | "terminus"
    a: str | None  # junction at the first end, None = dead end
    b: str | None  # junction at the second end, None = dead end
    cells: list[Cell]  # ordered a -> b, junction cells excluded
    # cell just outside each end: the junction cell, or None at a dead end
    a_cell: Cell | None = None
    b_cell: Cell | None = None

    @property
    def length(self) -> int:
        return len(self.cells)

    def other_end(self, junction: str | None) -> str | None:
        if junction == self.a and junction != self.b:
            return self.b
        if junction == self.b and junction != self.a:
            return self.a
        raise ValueError(f"{junction} is not a unique end of {self.name}")


@dataclass
class Junction:
    name: str
    cell: Cell
    stem: str  # section on the stem side
    branches: list[str]  # sections reachable from the stem

    def connected(self, s1: str, s2: str) -> bool:
        return (s1 == self.stem and s2 in self.branches) or (s2 == self.stem and s1 in self.branches)


@dataclass
class Layout:
    sections: dict[str, Section]
    junctions: dict[str, Junction]
    height: int
    width: int
    cell_index: dict[Cell, tuple[str, int]] = field(default_factory=dict)  # cell -> (section, idx)
    junction_at: dict[Cell, str] = field(default_factory=dict)

    # ---------------------------------------------------------------- build
    @classmethod
    def from_spec(cls, spec: dict) -> "Layout":
        junctions = {
            name: Junction(name, tuple(j["cell"]), j["stem"], list(j["branches"]))
            for name, j in spec["junctions"].items()
        }
        sections: dict[str, Section] = {}
        for s in spec["sections"]:
            path = rasterize([tuple(c) for c in s["path"]])
            a, b = s.get("a"), s.get("b")
            a_cell = b_cell = None
            if a is not None:
                if path[0] != junctions[a].cell:
                    raise ValueError(f"section {s['name']}: path must start on junction {a}")
                a_cell, path = path[0], path[1:]
            if b is not None:
                if path[-1] != junctions[b].cell:
                    raise ValueError(f"section {s['name']}: path must end on junction {b}")
                b_cell, path = path[-1], path[:-1]
            if len(path) < (3 if (a is None or b is None) else 1):
                raise ValueError(f"section {s['name']} is too short")
            sections[s["name"]] = Section(s["name"], s["kind"], a, b, path, a_cell, b_cell)

        all_cells = [c for s in sections.values() for c in s.cells] + [j.cell for j in junctions.values()]
        if len(all_cells) != len(set(all_cells)):
            raise ValueError("layout cells overlap")
        height = max(c[0] for c in all_cells) + 2
        width = max(c[1] for c in all_cells) + 2
        lay = cls(sections, junctions, height, width)
        for s in sections.values():
            for i, c in enumerate(s.cells):
                lay.cell_index[c] = (s.name, i)
        for j in junctions.values():
            lay.junction_at[j.cell] = j.name
            for name in [j.stem, *j.branches]:
                sec = sections[name]
                if j.name not in (sec.a, sec.b):
                    raise ValueError(f"junction {j.name} lists {name}, which does not end there")
        return lay

    def to_rail(self) -> RailGridTransitionMap:
        t = RailEnvTransitions()
        grid = np.zeros((self.height, self.width), dtype=np.uint16)

        def connect(cell: Cell, side_a: int, side_b: int) -> None:
            v = int(grid[cell])
            v = t.set_transition(v, (side_a + 2) % 4, side_b, 1)
            v = t.set_transition(v, (side_b + 2) % 4, side_a, 1)
            grid[cell] = v

        def dead_end(cell: Cell, open_side: int) -> None:
            grid[cell] = t.set_transition(int(grid[cell]), (open_side + 2) % 4, open_side, 1)

        for s in self.sections.values():
            chain = [s.a_cell, *s.cells, s.b_cell]
            for i in range(1, len(chain) - 1):
                prev, cur, nxt = chain[i - 1], chain[i], chain[i + 1]
                if prev is None and nxt is None:
                    raise ValueError(f"section {s.name} is isolated")
                if prev is None:
                    dead_end(cur, side_between(cur, nxt))
                elif nxt is None:
                    dead_end(cur, side_between(cur, prev))
                else:
                    connect(cur, side_between(cur, prev), side_between(cur, nxt))
        for j in self.junctions.values():
            stem_side = side_between(j.cell, self._adjacent_cell(j, j.stem))
            for br in j.branches:
                connect(j.cell, stem_side, side_between(j.cell, self._adjacent_cell(j, br)))
        rail = RailGridTransitionMap(width=self.width, height=self.height, transitions=t)
        rail.grid = grid
        return rail

    def _adjacent_cell(self, j: Junction, section: str) -> Cell:
        s = self.sections[section]
        ends = []
        if s.a == j.name:
            ends.append(s.cells[0])
        if s.b == j.name:
            ends.append(s.cells[-1])
        for c in ends:
            if abs(c[0] - j.cell[0]) + abs(c[1] - j.cell[1]) == 1:
                return c
        raise ValueError(f"section {section} does not touch junction {j.name}")

    # ------------------------------------------------------------- queries
    def junction_between(self, s1: str, s2: str) -> str | None:
        """Junction through which a train can pass directly from s1 to s2."""
        a, b = self.sections[s1], self.sections[s2]
        for jn in {a.a, a.b} & {b.a, b.b}:
            if jn is not None and self.junctions[jn].connected(s1, s2):
                return jn
        return None

    def start_config(self, section: str) -> tuple[Cell, int]:
        """Entry cell and heading for a train departing from an origin stub."""
        s = self.sections[section]
        if (s.a is None) == (s.b is None):
            raise ValueError(f"{section} is not a stub with exactly one dead end")
        cells = s.cells if s.a is None else s.cells[::-1]
        return cells[1], side_between(cells[1], cells[2] if len(cells) > 2 else (s.b_cell if s.a is None else s.a_cell))

    def target_cell(self, section: str) -> Cell:
        """Cell at which a train arriving in a terminus stub is complete."""
        s = self.sections[section]
        if (s.a is None) == (s.b is None):
            raise ValueError(f"{section} is not a stub with exactly one dead end")
        cells = s.cells if s.a is None else s.cells[::-1]
        return cells[1]

    def public(self) -> dict:
        """The network description shown to every controller."""
        return {
            "sections": [
                {"name": s.name, "kind": s.kind, "length_cells": s.length,
                 "ends": [s.a or "dead_end", s.b or "dead_end"]}
                for s in self.sections.values()
            ],
            "junctions": [
                {"name": j.name, "connects": [[j.stem, b] for b in j.branches]}
                for j in self.junctions.values()
            ],
        }
