"""Simple coded policies used as yardsticks. All act on the public observation only."""
from __future__ import annotations

from .reference import DONE, ReferenceDispatcher


def do_nothing(_obs: dict) -> list[dict]:
    return []


class GreedyFullRoutes:
    """Give every train its whole shortest route as soon as it may depart. No coordination."""

    def __init__(self, network: dict):
        self.ref = ReferenceDispatcher(network)

    def full_route(self, origin: str, dest: str) -> list[str]:
        route, sec, end = [origin], origin, self.ref.origin_exit(origin)
        while True:
            path, target = self.ref.hops(sec, end, dest)[0]
            route += path[1:]
            if target == DONE:
                return route
            sec, end = target

    def __call__(self, obs: dict) -> list[dict]:
        return [{"train": t["id"], "action": "route", "route": self.full_route(t["origin"], t["destination"])}
                for t in obs["trains"] if t["status"] == "ready" and not t["route"]]


class OneAtATime(GreedyFullRoutes):
    """Run trains strictly one after another, earliest deadline first.
    Always safe, never clever: a witness that the scenario is solvable, and a floor for 'slow but correct'."""

    def __call__(self, obs: dict) -> list[dict]:
        if any(t["route"] for t in obs["trains"] if t["status"] != "arrived"):
            return []
        ready = sorted((t for t in obs["trains"] if t["status"] == "ready"), key=lambda t: (t["deadline"], t["id"]))
        if not ready:
            return []
        t = ready[0]
        return [{"train": t["id"], "action": "route", "route": self.full_route(t["origin"], t["destination"])}]


def make(name: str, network: dict):
    if name == "nothing":
        return do_nothing
    if name == "greedy":
        return GreedyFullRoutes(network)
    if name == "serial":
        return OneAtATime(network)
    if name == "reference":
        return ReferenceDispatcher(network).decide
    raise ValueError(f"unknown policy {name!r}")
