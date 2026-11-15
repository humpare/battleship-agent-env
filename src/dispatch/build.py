"""Scenario construction helpers: a single-track line with passing loops."""
from __future__ import annotations


def linear_line(loops: int = 1, single_len: int = 7, loop_len: int = 7, stub_len: int = 5,
                siding_rows: int = 2) -> dict:
    """West terminal -- S1 -- loop L1 -- S2 -- ... -- East terminal.

    Each terminal has an origin stub (trains start there) and a terminus stub
    (trains finish there). Each loop has two parallel tracks, ``a`` (straight)
    and ``b`` (the siding, longer by 2 * siding_rows cells). Every single-track section
    between them is a bottleneck. A loop track holds as many trains as it has cells.
    """
    r = 5  # main line row
    junctions, sections = {}, []
    c = 1
    # west terminal
    jw = (r, c + stub_len)
    junctions["JW"] = {"cell": list(jw), "stem": "S1", "branches": ["W_dep", "W_arr"]}
    sections.append({"name": "W_dep", "kind": "origin", "a": None, "b": "JW", "path": [[r, c], list(jw)]})
    sections.append({"name": "W_arr", "kind": "terminus", "a": "JW", "b": None, "path": [list(jw), [r - 3, jw[1]]]})
    prev_j, col = "JW", jw[1]
    for k in range(1, loops + 1):
        jw_k, je_k = f"J{k}w", f"J{k}e"
        cw = col + single_len + 1
        ce = cw + loop_len + 1
        sections.append({"name": f"S{k}", "kind": "single", "a": prev_j, "b": jw_k, "path": [[r, col], [r, cw]]})
        junctions[jw_k] = {"cell": [r, cw], "stem": f"S{k}", "branches": [f"L{k}a", f"L{k}b"]}
        junctions[je_k] = {"cell": [r, ce], "stem": f"S{k + 1}", "branches": [f"L{k}a", f"L{k}b"]}
        sections.append({"name": f"L{k}a", "kind": "loop", "a": jw_k, "b": je_k, "path": [[r, cw], [r, ce]]})
        sections.append({"name": f"L{k}b", "kind": "loop", "a": jw_k, "b": je_k,
                         "path": [[r, cw], [r + siding_rows, cw], [r + siding_rows, ce], [r, ce]]})
        prev_j, col = je_k, ce
    ce = col + single_len + 1
    last = f"S{loops + 1}"
    sections.append({"name": last, "kind": "single", "a": prev_j, "b": "JE", "path": [[r, col], [r, ce]]})
    junctions["JE"] = {"cell": [r, ce], "stem": last, "branches": ["E_dep", "E_arr"]}
    sections.append({"name": "E_dep", "kind": "origin", "a": "JE", "b": None, "path": [[r, ce], [r, ce + stub_len]]})
    sections.append({"name": "E_arr", "kind": "terminus", "a": "JE", "b": None, "path": [[r, ce], [r - 3, ce]]})
    return {"junctions": junctions, "sections": sections}


def train(tid: str, direction: str, earliest: int, deadline: int) -> dict:
    """direction 'E' = west terminal to east terminal, 'W' = the reverse."""
    o, d = ("W_dep", "E_arr") if direction == "E" else ("E_dep", "W_arr")
    return {"id": tid, "origin": o, "destination": d, "earliest_departure": earliest, "deadline": deadline}


def free_run_ticks(layout: dict) -> int:
    """Ticks from the departure tick to arrival for a train that never waits (straight tracks)."""
    cells = 0
    for s in layout["sections"]:
        if s["kind"] == "single" or s["name"].endswith("a"):
            (r0, c0), (r1, c1) = s["path"][0], s["path"][-1]
            cells += abs(r1 - r0) + abs(c1 - c0)  # section cells plus one junction
    stub = next(s for s in layout["sections"] if s["name"] == "W_dep")
    return cells + abs(stub["path"][1][1] - stub["path"][0][1]) + 3


def make_scenario(sid: str, *, loops: int, n_trains: int, seed: int, single_len: int = 7, loop_len: int = 3,
                  headway: float = 4.0, slack: float = 1.6, malfunctions: int = 0, siding_rows: int = 2,
                  malfunction_ticks: tuple[int, int] = (8, 16),
                  stage: str = "", split: str = "dev", max_advance: int = 5) -> dict:
    """A reproducible scenario from a handful of knobs.

    headway: mean ticks between successive departure windows (lower = denser traffic).
    slack:   deadline = earliest_departure + slack * free-run time (lower = tighter).
    The horizon is the larger of 1.5x the latest deadline and the time a strictly
    one-train-at-a-time schedule would need, so every scenario is feasible by construction.
    """
    import random
    rng = random.Random(seed)
    layout = linear_line(loops, single_len, loop_len, siding_rows=siding_rows)
    free = free_run_ticks(layout)
    trains, t = [], 1
    for i in range(n_trains):
        d = rng.choice("EW") if i >= 2 else "EW"[i]  # always at least one train each way
        trains.append(train(f"T{i + 1}", d, t, t + int(round(slack * free))))
        t += max(0, int(round(rng.expovariate(1.0 / headway)))) if headway > 0 else 0
    events = []
    for _ in range(malfunctions):
        victim = rng.choice(trains)
        tick = victim["earliest_departure"] + rng.randint(6, max(7, free - 6))
        events.append({"tick": tick, "train": victim["id"], "duration": rng.randint(*malfunction_ticks)})
    # Feasible by construction: running the trains strictly one after another, with every
    # malfunction costing its full duration, still finishes inside the horizon.
    serial = max(x["earliest_departure"] for x in trains) + n_trains * (free + 2) + sum(e["duration"] for e in events)
    horizon = max(int(round(1.5 * max(x["deadline"] for x in trains))), serial) + 1
    return {"id": sid, "meta": {"stage": stage, "split": split, "seed": seed, "loops": loops,
                                "single_len": single_len, "loop_len": loop_len, "siding_rows": siding_rows,
                                "headway": headway,
                                "slack": slack, "free_run_ticks": free},
            "horizon": horizon, "max_advance": max_advance, "lateness_cap": free,
            "layout": layout, "trains": trains,
            "malfunctions": events}
