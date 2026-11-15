"""Simple controllers for the incident model, and exact evaluation of any controller.

A controller is a function (t, belief, history) -> action index, where history is the list of
(action name, observation) so far. Runbooks use only the history; belief-based ones use the belief.
"""
from __future__ import annotations

from functools import lru_cache

from .kernel import ACTIONS, Params, initial_belief
from .solver import expand, step

A = {n: i for i, n in enumerate(ACTIONS)}


def evaluate(p: Params, controller) -> float:
    """Exact expected return of a controller (walks every observation branch)."""
    def go(t, b, hist):
        if t >= p.horizon:
            return 0.0
        a = controller(t, b, hist)
        r, succ = expand(p, b, a)
        return r + sum(z * go(t + 1, b2, hist + [(ACTIONS[a], obs)]) for obs, (z, b2) in succ.items())
    return go(0, initial_belief(p), [])


# ------------------------------------------------------------------ runbooks (history only)
def idle(t, b, hist):
    return A["wait"]


def _healthy(hist):
    return bool(hist) and hist[-1][1][1]


def _known(hist):
    """What the runbook believes about (fenced, promoted, routed, lag): 1/0 if confirmed, None if not."""
    F = P = R = lag = None
    for name, (obs, _) in hist:
        if isinstance(obs, tuple) and obs[0] == "inspect":
            _, F, P, R, lag = obs
        elif obs == "ack":
            if name == "fence_a":
                F = 1
            elif name == "promote_b":
                P = 1
            elif name == "route_b":
                R = 1
    return F, P, R, lag


def make_runbook(verify: bool, wait_first: int = 0, check_lag: bool = True, restart_if_lag: bool = True):
    """Fail over to the replica: fence, promote, route, in that order.
    verify: after a timeout, inspect before going on (otherwise simply re-issue until acknowledged).
    wait_first: ticks to wait before starting, in case the primary recovers on its own.
    check_lag / restart_if_lag: inspect the replica first and try a restart when it is behind."""
    def run(t, b, hist):
        if _healthy(hist):
            return A["wait"]
        if t < wait_first:
            return A["wait"]
        F, P, R, lag = _known(hist)
        names = [h[0] for h in hist]
        if check_lag and lag is None:
            return A["inspect"]
        if restart_if_lag and lag == 1 and not F:
            if "restart_a" not in names:
                return A["restart_a"]
            if names.count("restart_a") == 1 and len(names) - 1 - names.index("restart_a") < 2:
                return A["wait"]            # restart still running
        for flag, cmd in ((F, "fence_a"), (P, "promote_b"), (R, "route_b")):
            if not flag:
                if verify and hist and hist[-1][0] == cmd and hist[-1][1][0] == "timeout":
                    return A["inspect"]
                return A[cmd]
        return A["wait"]
    return run


# ------------------------------------------------------------------ belief-based approximations
def make_mdp(p: Params):
    @lru_cache(maxsize=None)
    def q(t, s, a):
        if t >= p.horizon:
            return 0.0
        return sum(pr * (r + v(t + 1, s2)) for pr, s2, _, r in step(p, s, a))

    @lru_cache(maxsize=None)
    def v(t, s):
        if t >= p.horizon:
            return 0.0
        return max(q(t, s, a) for a in range(len(ACTIONS)))
    return q, v


def make_qmdp(p: Params):
    """Act as if everything hidden were revealed after this action (never values information)."""
    q, _ = make_mdp(p)

    def run(t, b, hist):
        return max(range(len(ACTIONS)), key=lambda a: sum(pb * q(t, s, a) for s, pb in b.items()))
    return run


def make_lookahead(p: Params, depth: int):
    """Exact belief search for `depth` ticks, then the QMDP estimate."""
    q, v = make_mdp(p)

    def val(t, b, d):
        if t >= p.horizon:
            return 0.0
        if d == 0:
            return max(sum(pb * q(t, s, a) for s, pb in b.items()) for a in range(len(ACTIONS)))
        return max(qa(t, b, a, d) for a in range(len(ACTIONS)))

    def qa(t, b, a, d):
        r, succ = expand(p, b, a)
        return r + sum(z * val(t + 1, b2, d - 1) for z, b2 in succ.values())

    def run(t, b, hist):
        return max(range(len(ACTIONS)), key=lambda a: qa(t, b, a, depth))
    return run


def make_exact(solver):
    def run(t, b, hist):
        qs = solver.q(t, b)
        return max(range(len(ACTIONS)), key=lambda a: qs[a])
    return run
