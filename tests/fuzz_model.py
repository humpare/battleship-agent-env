"""The public forward model must predict the real simulator exactly (given the same
commands and no malfunctions beyond those already visible when it starts).

Usage: uv run python tests/fuzz_model.py [episodes]
"""
import importlib.util
import pathlib
import random
import sys

from dispatch.build import make_scenario
from dispatch.model import PublicModel
from dispatch.reference import ReferenceDispatcher
from dispatch.rules import rules
from dispatch.service import EpisodeService

spec = importlib.util.spec_from_file_location("fz", pathlib.Path(__file__).with_name("fuzz_random_play.py"))
fz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fz)
KEYS = ("id", "status", "location", "route", "holding", "waiting_at_end_of_route", "blocked_by",
        "malfunction_ticks_remaining", "arrived_at")


def view(obs):
    return (obs["time"], obs["decision_id"], obs["episode_over"],
            [{k: t[k] for k in KEYS} for t in obs["trains"]],
            [{k: v for k, v in e.items() if k != "ticks_remaining"} for e in obs["events_since_last_decision"]])


def main(n: int, with_malfunctions: bool = False) -> None:
    rng = random.Random(3)
    checked = 0
    for ep in range(n):
        sc = make_scenario(f"m{ep}", loops=rng.randint(1, 3), n_trains=rng.randint(2, 10), seed=ep,
                           single_len=rng.randint(4, 8), loop_len=rng.randint(1, 4), siding_rows=rng.choice([1, 2]),
                           headway=rng.choice([0, 2, 5]), malfunctions=rng.randint(1, 3) if with_malfunctions else 0)
        svc = EpisodeService(sc)
        r = svc.get_rules()
        ref = ReferenceDispatcher(r["network"])
        random_play = ep % 2 == 1
        model = PublicModel(r, svc.observe())
        k = 0
        while not svc.sim.over:
            obs = svc.observe()
            cmds = fz.random_commands(rng, svc) if random_play else ref.decide(obs)
            ans = svc.step(cmds, obs["decision_id"], f"r{k}")
            k += 1
            if ans["status"] != "accepted":
                cmds = []
                ans = svc.step(cmds, obs["decision_id"], f"r{k}")
                k += 1
            model.apply(cmds)
            model.advance()
            real = ans["observation"]
            if with_malfunctions and any(e["type"] == "malfunction_started" for e in real["events_since_last_decision"]):
                model = PublicModel(r, real)  # an unforeseeable event: re-sync, as the reference does
                continue
            a, b = view(real), view(model.observation())
            if a != b:
                print(f"MISMATCH episode {ep} ({'random' if random_play else 'reference'}) at tick {real['time']}")
                print(" real  time/dec/over", a[:3], "\n model time/dec/over", b[:3])
                for x, y in zip(a[3], b[3]):
                    if x != y:
                        print("  real ", x, "\n  model", y)
                print("  real events ", a[4], "\n  model events", b[4])
                sys.exit(1)
            checked += 1
        s_model = model.score()
        res = svc.sim.result()
        assert abs(s_model[0] - res["score"]) < 1e-6 and s_model[1] == res["completed"], (ep, s_model, res["score"])
    print(f"{n} episodes, {checked} decisions: model == simulator" + (" (with malfunctions)" if with_malfunctions else ""))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, len(sys.argv) > 2)
