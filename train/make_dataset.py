"""Build a behaviour-cloning dataset from the environment.

Each example is one decision: (rules + network + observation) -> the commands the coded
reference issued. Scenarios are drawn from the same generator as the benchmark, with seeds
disjoint from the dev and held-out seeds.

Usage: uv run python train/make_dataset.py --episodes 600 --out train/data/bc.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from dispatch.build import make_scenario
from dispatch.ladder import SEEDS
from dispatch.llm_runner import stateless_messages
from dispatch.reference import ReferenceDispatcher
from dispatch.rules import rules
from dispatch.sim import DispatchSim

RESERVED = {s for seeds in SEEDS.values() for s in seeds}


def sample_knobs(rng: random.Random) -> dict:
    return dict(loops=rng.randint(1, 3), n_trains=rng.randint(2, 16), single_len=rng.randint(5, 9),
                loop_len=rng.choice([2, 3, 4]), headway=rng.choice([0, 2, 4, 6]),
                slack=rng.choice([1.5, 1.6, 2.0, 3.0]), malfunctions=rng.choice([0, 0, 1, 2, 3]))


def episode_examples(scenario: dict):
    sim = DispatchSim(scenario)
    r = rules(sim)
    ref = ReferenceDispatcher(r["network"])
    while not sim.over:
        obs = sim.observation()
        commands = ref.decide(obs)
        yield stateless_messages(r, obs), commands
        sim.apply(commands)
        sim.advance()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--first-seed", type=int, default=10_000)
    ap.add_argument("--keep-empty", type=float, default=0.5, help="fraction of 'no command' decisions kept")
    ap.add_argument("--out", default="train/data/bc.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.first_seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = empty = 0
    with out.open("w") as f:
        for k in range(a.episodes):
            seed = a.first_seed + k
            assert seed not in RESERVED
            sc = make_scenario(f"train-{seed}", seed=seed, **sample_knobs(rng))
            for messages, commands in episode_examples(sc):
                if not commands and rng.random() > a.keep_empty:
                    continue
                empty += not commands
                n += 1
                f.write(json.dumps({"messages": messages + [
                    {"role": "assistant", "content": json.dumps(commands, separators=(",", ":"))}]}) + "\n")
    print(f"{n} examples ({empty} with no command) from {a.episodes} episodes -> {out}")


if __name__ == "__main__":
    main()
