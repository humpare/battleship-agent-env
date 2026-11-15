"""The scenario family: one generator, six difficulty stages, dev and held-out seeds.

Usage: python -m dispatch.ladder   (rewrites scenarios/dev and scenarios/heldout)
"""
from __future__ import annotations

import json
from pathlib import Path

from .build import make_scenario

ROOT = Path(__file__).resolve().parents[2]

# One major factor changes per stage. Scoring rules never change.
STAGES = {
    "a-basic":       dict(loops=1, n_trains=4,  headway=6, slack=3.0, loop_len=3, malfunctions=0),
    "b-interaction": dict(loops=2, n_trains=6,  headway=4, slack=2.0, loop_len=3, malfunctions=0),
    "c-pressure":    dict(loops=2, n_trains=6,  headway=2, slack=1.5, loop_len=3, malfunctions=0),
    "d-density":     dict(loops=3, n_trains=10, headway=2, slack=1.6, loop_len=2, malfunctions=0),
    "e-recovery":    dict(loops=3, n_trains=10, headway=2, slack=1.6, loop_len=2, malfunctions=2),
    "f-sustained":   dict(loops=3, n_trains=16, headway=5, slack=1.6, loop_len=2, malfunctions=3),
}
SEEDS = {"dev": [1], "heldout": [101, 102]}


def build_all() -> list[dict]:
    out = []
    for split, seeds in SEEDS.items():
        for stage, knobs in STAGES.items():
            for seed in seeds:
                out.append(make_scenario(f"{stage}-s{seed}", seed=seed, stage=stage, split=split, **knobs))
    return out


def main() -> None:
    for sc in build_all():
        d = ROOT / "scenarios" / sc["meta"]["split"]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sc['id']}.json").write_text(json.dumps(sc, indent=1))
        print(f"{sc['meta']['split']:8} {sc['id']:22} trains={len(sc['trains']):2} horizon={sc['horizon']:4} "
              f"malfunctions={len(sc['malfunctions'])}")


if __name__ == "__main__":
    main()
