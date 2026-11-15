"""Random-play fuzz: arbitrary (valid and invalid) command streams must never crash the
simulator, never produce a score outside [0, 1], and always replay to the identical result.

Usage: uv run python tests/fuzz_random_play.py [episodes]
"""
import random
import sys

from dispatch.build import make_scenario
from dispatch.service import EpisodeService
from dispatch.sim import replay

JUNK = [None, 5, "x", {"train": "T1"}, {"train": "T1", "action": "route", "route": []},
        {"train": "T1", "action": "route", "route": [["S1"]]}, {"train": ["T1"], "action": "hold"},
        {"train": "T1", "action": "hold", "route": ["S1"]}, {"train": "T1", "action": "route", "route": "S1"},
        {"train": "T1", "action": "route", "route": ["S1"], "extra": 1}, [], {"action": "hold"}]


def random_commands(rng, svc):
    sim = svc.sim
    names = list(sim.layout.sections)
    out = []
    for tr in sim.trains:
        roll = rng.random()
        if tr.arrival_tick is not None or roll < 0.55:
            continue
        if roll < 0.65:
            out.append({"train": tr.id, "action": "hold"})
            continue
        start = tr.route[0] if tr.route else tr.origin
        if roll < 0.75:  # arbitrary section soup: usually illegal
            cand = [start] + rng.sample(names, rng.randint(0, 3))
        else:  # random walk that respects connectivity and direction
            cand, sec, end = [start], start, sim._heading_end(tr)
            for _ in range(rng.randint(0, 6)):
                if end is None:
                    break
                j = sim.layout.junctions[end]
                options = [s for s in (j.stem, *j.branches) if j.connected(sec, s) and s not in cand]
                if not options:
                    break
                nxt = rng.choice(options)
                cand.append(nxt)
                end, sec = sim.layout.sections[nxt].other_end(end), nxt
        out.append({"train": tr.id, "action": "route", "route": cand})
    if rng.random() < 0.05:
        out.append(rng.choice(JUNK))
    return out


def main(n: int) -> None:
    rng = random.Random(1)
    stats = {"accepted": 0, "refused": 0, "arrivals": 0, "deadlocked": 0}
    for ep in range(n):
        sc = make_scenario(f"fz{ep}", loops=rng.randint(1, 3), n_trains=rng.randint(2, 10), seed=ep,
                           single_len=rng.randint(4, 8), loop_len=rng.randint(2, 4),
                           headway=rng.choice([0, 2, 5]), malfunctions=rng.randint(0, 3))
        svc = EpisodeService(sc)
        k = 0
        while not svc.sim.over:
            batch = random_commands(rng, svc)
            ans = svc.step(batch, svc.sim.decision_id, f"r{k}")
            k += 1
            if ans["status"] == "accepted":
                stats["accepted"] += 1
            else:
                stats["refused"] += 1
                assert not ans["time_advanced"]
                svc.step([], svc.sim.decision_id, f"r{k}e")
                k += 1
        live = svc.sim.result()
        again = replay(sc, svc.batches)
        assert again["per_train"] == live["per_train"] and again["score"] == live["score"], f"replay mismatch in {ep}"
        assert 0.0 <= live["score"] <= 1.0
        stats["arrivals"] += live["completed"]
        stats["deadlocked"] += len(live["diagnostics"]["deadlocked_trains"])
    print(f"{n} random episodes ok: {stats}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
