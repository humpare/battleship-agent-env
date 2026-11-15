"""Lookahead reference: the coded dispatcher that plays ahead before it commits.

At every decision it takes several complete strategies (the safe hop-by-hop dispatcher under
different priority rules), plays each one to the end of the episode in the public forward
model, and issues the first move of whichever ends best. It re-plans from scratch at the next
decision, so a malfunction is taken into account as soon as it is visible.

Public data only: rules, network, timetable, observation. No simulator access.
"""
from __future__ import annotations

try:
    from .model import PublicModel
    from .reference import ReferenceDispatcher
except ImportError:  # run as plain scripts from solution/
    from model import PublicModel
    from reference import ReferenceDispatcher


class LookaheadDispatcher:
    def __init__(self, rules: dict, priorities: list[str] | None = None):
        self.rules = rules
        if priorities is None:
            termini = [s["name"] for s in rules["network"]["sections"] if s["kind"] == "terminus"]
            priorities = ["slack", "departure", "deadline"] + [f"to:{t}" for t in termini]
        self.variants = [(p, ReferenceDispatcher(rules["network"], priority=p)) for p in priorities]
        self.chosen: dict[str, int] = {}
        self.rollouts = 0

    def _play_out(self, policy: ReferenceDispatcher, obs: dict, first: list[dict]) -> tuple:
        m = PublicModel(self.rules, obs)
        m.apply(first)
        while not m.over:
            m.advance()
            if m.over:
                break
            m.apply(policy.decide(m.observation()))
        score, done, late = m.score()
        return (round(score, 9), done, -late)

    def decide(self, obs: dict) -> list[dict]:
        best = None
        seen: dict[str, tuple] = {}
        for name, policy in self.variants:
            first = policy.decide(obs)
            key = repr(sorted((c["train"], tuple(c.get("route", ()))) for c in first)) + name
            outcome = self._play_out(policy, obs, first)
            self.rollouts += 1
            seen[key] = outcome
            if best is None or outcome > best[0]:
                best = (outcome, name, first)
        self.chosen[best[1]] = self.chosen.get(best[1], 0) + 1
        return best[2]
