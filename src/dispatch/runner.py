"""In-process episode runner (no MCP): used for tests, calibration and replay."""
from __future__ import annotations

import time

from .reference import ReferenceDispatcher
from .rules import rules
from .sim import DispatchSim


def run_policy(scenario: dict, policy=None, max_decisions: int | None = None) -> tuple[dict, list]:
    """Run ``policy(obs) -> commands`` to the end. Returns (result, action log)."""
    sim = DispatchSim(scenario)
    if policy is None:
        policy = ReferenceDispatcher(rules(sim)["network"]).decide
    log, t0 = [], time.perf_counter()
    while not sim.over and (max_decisions is None or len(log) < max_decisions):
        commands = policy(sim.observation())
        sim.apply(commands)
        log.append(commands)
        sim.advance()
    sim.run_out()
    res = sim.result()
    res["policy_seconds"] = round(time.perf_counter() - t0, 3)
    return res, log
