"""Coded reference dispatcher.

Pure Python over the *public* data only: the network from ``get_rules`` and the
observation from ``observe`` / ``step``. It imports nothing from the simulator.

Policy: move trains from one holding place (origin, passing-loop track) to the
next, one "hop" at a time. A hop is granted only when

1. no train claims any section on the way in the opposite direction, and no
   train is parked on the way;
2. the holding track at the end of the hop has room: a loop track holds as many
   trains as it has cells, provided they all travel the same way to the same
   destination (so they are interchangeable and queue order cannot matter);
3. the resulting state is *safe*: the number of trains that can still be
   brought home (found by exhaustive search over hop orders, Banker's-algorithm
   style) does not drop.

Among trains competing for track, the one with the least slack against its
deadline goes first. A train that is only waiting for opposing traffic to clear
keeps its claim, so a stream of trains in the other direction cannot starve it.
"""
from __future__ import annotations

DEAD = "dead_end"
DONE = "DONE"


class ReferenceDispatcher:
    def __init__(self, network: dict, node_budget: int = 200_000, priority: str = "slack"):
        """priority: how competing trains are ordered.
        "slack" (least slack against the deadline first), "departure" (earliest departure window
        first), "deadline", or "to:<terminus>" (trains bound for that terminus first, then slack)."""
        self.priority = priority
        self.sec = {s["name"]: s for s in network["sections"]}
        self.adj: dict[tuple[str, str], list[str]] = {}
        for j in network["junctions"]:
            for a, b in j["connects"]:
                self.adj.setdefault((a, j["name"]), []).append(b)
                self.adj.setdefault((b, j["name"]), []).append(a)
        self.node_budget = node_budget
        self._hop_cache: dict = {}
        self._dist_cache: dict = {}

    # ------------------------------------------------------------ topology
    def other_end(self, section: str, end: str) -> str:
        a, b = self.sec[section]["ends"]
        return b if end == a else a

    def origin_exit(self, section: str) -> str:
        a, b = self.sec[section]["ends"]
        return a if a != DEAD else b

    def hops(self, section: str, end: str, dest: str):
        """Paths from ``section`` (leaving by ``end``) to the next holding track
        or to the destination. Returns [(path, target)], target = (section, end) or DONE."""
        key = (section, end, dest)
        if key in self._hop_cache:
            return self._hop_cache[key]
        out = []

        def walk(sec, e, path):
            for nxt in self.adj.get((sec, e), []):
                if nxt in path:
                    continue
                kind = self.sec[nxt]["kind"]
                if kind == "origin" or (kind == "terminus" and nxt != dest):
                    continue
                if kind == "terminus":
                    out.append((path + [nxt], DONE))
                elif kind == "loop":
                    out.append((path + [nxt], (nxt, self.other_end(nxt, e))))
                else:
                    walk(nxt, self.other_end(nxt, e), path + [nxt])

        if end != DEAD:
            walk(section, end, [section])
        out.sort(key=lambda h: (sum(self.sec[s]["length_cells"] for s in h[0]), h[0]))
        self._hop_cache[key] = out
        return out

    def dist_to_dest(self, section: str, end: str, dest: str) -> int:
        """Fewest cells from the far end of ``section`` to the destination."""
        key = (section, end, dest)
        if key not in self._dist_cache:
            best = 10 ** 6
            for path, target in self.hops(section, end, dest):
                d = sum(self.sec[s]["length_cells"] + 1 for s in path[1:])
                if target != DONE:
                    d += self.dist_to_dest(target[0], target[1], dest)
                best = min(best, d)
            self._dist_cache[key] = best
        return self._dist_cache[key]

    # ------------------------------------------------------ abstract state
    def _walk_route(self, route: list[str], first_end: str) -> list[tuple[str, str]]:
        """(section, exit end) for every section of a route."""
        out, end = [], first_end
        for i, s in enumerate(route):
            if i > 0:
                end = self.other_end(s, end)  # entered through the previous exit junction
            out.append((s, end))
        return out

    def cap(self, section: str) -> int:
        return self.sec[section]["length_cells"] if self.sec[section]["kind"] == "loop" else 1

    def _room(self, resting: dict, section: str, end: str, dest: str) -> bool:
        """May one more train (leaving by ``end`` for ``dest``) rest in ``section``?"""
        there = resting.get(section, ())
        return len(there) < self.cap(section) and all(x == (end, dest) for x in there)

    def value(self, items: tuple, memo: dict, budget: list) -> int:
        """Most trains that can still reach their destination from this state.
        ``items``: sorted tuple of (section, exit_end, dest) for unfinished trains."""
        if not items:
            return 0
        if items in memo:
            return memo[items]
        budget[0] -= 1
        if budget[0] < 0:
            return 0  # out of budget: assume the worst
        resting: dict = {}
        for sec, end, dest in items:
            resting.setdefault(sec, []).append((end, dest))
        best = 0
        tried = set()
        for i, it in enumerate(items):
            if it in tried:
                continue
            tried.add(it)
            sec, end, dest = it
            rest = items[:i] + items[i + 1:]
            for path, target in self.hops(sec.lstrip("@"), end, dest):
                if target == DONE:
                    if any(s in resting for s in path[1:]):
                        continue
                    v = 1 + self.value(rest, memo, budget)
                    memo[items] = v  # finishing a train only frees track: take it
                    return v
                if any(s in resting for s in path[1:-1]) or not self._room(resting, target[0], target[1], dest):
                    continue
                v = self.value(tuple(sorted(rest + ((target[0], target[1], dest),))), memo, budget)
                best = max(best, v)
                if best == len(items):
                    memo[items] = best
                    return best
        memo[items] = best
        return best

    # -------------------------------------------------------------- decide
    def decide(self, obs: dict) -> list[dict]:
        now = obs["time"]
        claims: set[tuple[str, str]] = set()  # (section, exit end) wanted or held by some train
        resting: dict[str, list] = {}  # section -> [(exit end, dest)] of trains that will stop there
        state: dict[str, tuple] = {}  # train -> abstract item (None once committed home)
        cands = []
        for t in obs["trains"]:
            if t["status"] == "arrived":
                continue
            dest, loc, route = t["destination"], t["location"], t["route"]
            if loc is None and not route:
                end = self.origin_exit(t["origin"])
                state[t["id"]] = ("@" + t["origin"], end, dest)
                if t["status"] != "scheduled":
                    cands.append((t, t["origin"], end, [t["origin"]]))
                continue
            first_end = loc["heading_to"] if loc is not None else self.origin_exit(route[0])
            if t["holding"]:
                route = route[:1]
            legs = self._walk_route(route, first_end)
            claims.update(legs)
            last, last_end = legs[-1]
            if self.sec[last]["kind"] == "terminus":
                state[t["id"]] = None
                continue
            resting.setdefault(last, []).append((last_end, dest))
            state[t["id"]] = (last, last_end, dest)
            if not t["holding"] and len(route) == 1:  # extend only from the last authorised section
                cands.append((t, last, last_end, route))

        def slack(c):
            t, sec, end, route = c
            remaining = self.dist_to_dest(sec, end, t["destination"])
            if t["location"] is not None:
                remaining += t["location"].get("cells_to_section_end", 0)
            else:
                remaining += self.sec[t["origin"]]["length_cells"]
            return t["deadline"] - (now + remaining)

        def behind_resting_train(c, granted):
            """A train queued behind one that has no onward route cannot leave first."""
            t, sec, end, _ = c
            if t["location"] is None:
                return False
            mine = t["location"].get("cells_to_section_end", 0)
            for o in obs["trains"]:
                ol = o["location"]
                if o["id"] == t["id"] or ol is None or o["id"] in granted:
                    continue
                if ol["section"] == sec and ol["heading_to"] == end and len(o["route"]) == 1 \
                        and ol.get("cells_to_section_end", 0) < mine:
                    return True
            return False

        def rank(c):
            t = c[0]
            if self.priority == "departure":
                return (t["earliest_departure"], slack(c), t["id"])
            if self.priority == "deadline":
                return (t["deadline"], slack(c), t["id"])
            if self.priority.startswith("to:"):
                return (t["destination"] != self.priority[3:], slack(c), t["id"])
            return (slack(c), t["id"])

        cands.sort(key=rank)
        memo: dict = {}
        budget = [self.node_budget]
        base_value = self.value(self._strip(state), memo, budget)
        commands, granted = [], set()
        queue, retried = list(cands), set()
        while queue:
            c = queue.pop(0)
            t, sec, end, route = c
            if behind_resting_train(c, granted):
                if t["id"] not in retried:  # the train ahead may be granted later this round
                    retried.add(t["id"])
                    queue.append(c)
                continue
            pending = None
            for path, target in self.hops(sec, end, t["destination"]):
                if any(s in resting for s in (path[1:] if target == DONE else path[1:-1])):
                    continue
                if target != DONE and not self._room(resting, target[0], target[1], t["destination"]):
                    continue
                trial = dict(state)
                trial[t["id"]] = None if target == DONE else (target[0], target[1], t["destination"])
                gain = 1 if target == DONE else 0
                if gain + self.value(self._strip(trial), memo, budget) < base_value:
                    continue
                legs = self._walk_route(path, end)
                if any((s, self.other_end(s, e)) in claims for s, e in legs[1:]):
                    pending = pending or (trial, legs, target, gain, False)
                    continue
                commands.append({"train": t["id"], "action": "route", "route": route + path[1:]})
                pending = (trial, legs, target, gain, True)
                break
            if pending is None:
                continue
            # granted, or waiting for opposing traffic with its claim kept
            trial, legs, target, gain, is_granted = pending
            state = trial
            base_value -= gain
            claims.update(legs[1:])
            if target != DONE:
                resting.setdefault(target[0], []).append((target[1], t["destination"]))
            if is_granted:
                granted.add(t["id"])
                if sec in resting and (end, t["destination"]) in resting[sec]:
                    resting[sec].remove((end, t["destination"]))
                    if not resting[sec]:
                        del resting[sec]
        return commands

    @staticmethod
    def _strip(state: dict) -> tuple:
        """Canonical search state. Trains not yet on the network sit on a virtual
        section ("@origin") that blocks nobody."""
        return tuple(sorted(s for s in state.values() if s))
