"""Data point: the published Flatland deadlock-avoidance heuristic on our scenarios.

It is NOT the benchmark's reference: it reads the simulator object directly (grid, agent
positions, per-cell paths) and issues native per-cell actions, so it does not go through
the public observation or the command interface. It is reported separately, as an
"offline policy with full simulator access".

Usage: uv run --with "flatland-baselines @ git+https://github.com/flatland-association/flatland-baselines@c1733cb" \
           python scripts/published_baseline.py [scenario.json ...]
"""
import glob
import json
import sys
import warnings

warnings.simplefilter("ignore")
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy  # noqa: E402

from dispatch.sim import DispatchSim  # noqa: E402

for path in sys.argv[1:] or sorted(glob.glob("scenarios/*/*.json")):
    sc = json.load(open(path))
    sim = DispatchSim(sc)  # builds the identical Flatland env, including scheduled malfunctions
    env = sim.env
    try:
        policy = DeadLockAvoidancePolicy(env=env) if "env" in DeadLockAvoidancePolicy.__init__.__code__.co_varnames else DeadLockAvoidancePolicy()
        handles = env.get_agent_handles()
        arrival = {}
        while env._elapsed_steps < sim.horizon and len(arrival) < len(handles):
            actions = policy.act_many(handles, observations=[None] * len(handles))
            env.step(actions)
            for a in env.agents:
                if a.arrival_time is not None and a.handle not in arrival:
                    arrival[a.handle] = a.arrival_time
        score = 0.0
        for tr in sim.trains:
            if tr.handle in arrival:
                late = max(0, arrival[tr.handle] - tr.deadline)
                score += 1 - 0.5 * min(late, sim.lateness_cap) / sim.lateness_cap
        print(f"{sc['id']:22} published-baseline score={score / len(sim.trains):.3f} completed={len(arrival)}/{len(handles)}")
    except Exception as e:
        print(f"{sc['id']:22} published-baseline FAILED: {type(e).__name__}: {str(e)[:160]}")
