"""Forward model of the railway, built ONLY from what every controller is given:
the rules, the network and timetable from get_rules(), and one observation.

It imports nothing from the simulator. The coded reference uses it to look ahead
("if I issue these orders, what happens?"). It cannot know future malfunctions:
it only carries forward the ones visible in the observation it starts from.
"""
from __future__ import annotations

DEAD = "dead_end"


class _T:
    __slots__ = ("id", "h", "origin", "dest", "ed", "deadline", "route", "hold", "sec", "k", "end", "junc",
                 "on", "arrived", "malf", "blocked_by", "at_end")


class PublicModel:
    def __init__(self, rules: dict, obs: dict):
        net = rules["network"]
        self.sec = {s["name"]: s for s in net["sections"]}
        self.length = {n: s["length_cells"] for n, s in self.sec.items()}
        self.pairs: dict[str, set] = {}
        for j in net["junctions"]:
            self.pairs[j["name"]] = {frozenset(p) for p in j["connects"]}
        self.horizon = rules["horizon"]
        self.max_advance = rules["max_advance_ticks"]
        self.cap = rules["lateness_cap_ticks"]
        self.time = obs["time"]
        self.decision_id = obs["decision_id"]
        self.events: list[dict] = []
        self.deadlocked: set[str] = set()
        self.trains: list[_T] = []
        order = {t["train"]: i for i, t in enumerate(rules["timetable"])}
        for o in sorted(obs["trains"], key=lambda o: order[o["id"]]):
            t = _T()
            t.id, t.h = o["id"], order[o["id"]]
            t.origin, t.dest = o["origin"], o["destination"]
            t.ed, t.deadline = o["earliest_departure"], o["deadline"]
            t.route, t.hold = list(o["route"]), o["holding"]
            t.arrived = o["arrived_at"]
            t.malf = o["malfunction_ticks_remaining"]
            t.blocked_by, t.at_end = o["blocked_by"], o["waiting_at_end_of_route"]
            loc = o["location"]
            t.on = loc is not None
            t.sec = t.k = t.end = t.junc = None
            if loc is not None:
                t.sec, t.end = loc["section"], loc["heading_to"]
                t.junc = loc.get("on_junction")
                t.k = loc["cells_to_section_end"]
            self.trains.append(t)
        self.by_id = {t.id: t for t in self.trains}
        self.deadlocked = set(self._deadlocked())  # already announced before this observation

    # ------------------------------------------------------------ geometry
    def other_end(self, section: str, end: str) -> str:
        a, b = self.sec[section]["ends"]
        return b if end == a else a

    def _cell(self, t: _T):
        if not t.on:
            return None
        if t.junc:
            return ("J", t.junc)
        ends = self.sec[t.sec]["ends"]
        return (t.sec, self.length[t.sec] - 1 - t.k if t.end == ends[1] else t.k)

    def _first_cell(self, section: str, junction: str):
        """Cell of ``section`` adjacent to ``junction``."""
        ends = self.sec[section]["ends"]
        return (section, 0 if junction == ends[0] else self.length[section] - 1)

    def _origin_exit(self, section: str) -> str:
        a, b = self.sec[section]["ends"]
        return a if a != DEAD else b

    def _want(self, t: _T):
        """Cell the train tries to enter this tick (None = stays), mirroring the command rules."""
        if t.arrived is not None or t.hold:
            return None
        if not t.on:
            if not t.route or self.time + 1 < t.ed:
                return None
            end = self._origin_exit(t.origin)
            ends = self.sec[t.origin]["ends"]
            k = self.length[t.origin] - 2
            return (t.origin, self.length[t.origin] - 1 - k if end == ends[1] else k)
        if t.junc:
            return self._first_cell(t.route[0], t.junc)
        if t.k > 0:
            ends = self.sec[t.sec]["ends"]
            k = t.k - 1
            return (t.sec, self.length[t.sec] - 1 - k if t.end == ends[1] else k)
        if t.end == DEAD or len(t.route) < 2:
            return None
        return ("J", t.end)

    # ---------------------------------------------------------------- tick
    def tick(self) -> list[dict]:
        events: list[dict] = []
        now = self.time + 1
        active = [t for t in self.trains if t.arrived is None]
        pos = {t.id: self._cell(t) for t in active}
        occupant = {c: tid for tid, c in pos.items() if c is not None}
        want = {t.id: self._want(t) for t in active}
        stuck = {t.id: t.malf > 0 for t in active}  # broken down during this tick
        ready = {t.id: (not t.on) and self.time >= t.ed for t in active}
        # --- conflict resolution, a port of Flatland's MotionCheck
        agents, reverse = {}, {}
        for t in active:
            r1 = pos[t.id] if pos[t.id] is not None else (None, t.h)
            can_try = want[t.id] is not None and not stuck[t.id] and (t.on or ready[t.id])
            r2 = want[t.id] if can_try else r1
            agents[t.h] = (r1, r2)
            reverse.setdefault(r2, []).append(t.h)
        stopped, dead, conflicts = set(), set(), []
        for i, (p, tg) in agents.items():
            for a2 in reverse.get(p, ()):
                if i < a2:
                    p2, t2 = agents[a2]
                    if p == t2 and tg == p2:
                        stopped.update((i, a2))
                        dead.update((i, a2))
            for a2 in reverse[tg]:
                if i < a2:
                    conflicts.append((i, a2))
            if p == tg:
                stopped.add(i)

        def stop(v):
            vp, vt = agents[v]
            agents[v] = (vp, vp)
            stopped.add(v)
            if vp in reverse:
                conflicts.extend((v, o) if v < o else (o, v) for o in reverse[vp] if o != v)
                reverse[vp].append(v)
            else:
                reverse[vp] = [v]
            reverse[vt].remove(v)

        for a in sorted(dead):
            stop(a)
        qi = 0
        while qi < len(conflicts):
            u, v = conflicts[qi]
            qi += 1
            up, ut = agents[u]
            vp, vt = agents[v]
            if vp == vt:
                if up != ut:
                    stop(u)
            elif ut == vt:
                stop(v)
        # --- apply
        for t in active:
            moved = False
            if want[t.id] is not None and t.h not in stopped and agents[t.h][0] != agents[t.h][1]:
                moved = True
                tgt = want[t.id]
                if not t.on:
                    t.on, t.sec, t.junc = True, t.origin, None
                    t.end, t.k = self._origin_exit(t.origin), self.length[t.origin] - 2
                elif t.junc:
                    j = t.junc
                    t.junc, t.sec = None, t.route[0]
                    t.end, t.k = self.other_end(t.sec, j), self.length[t.sec] - 1
                elif tgt[0] == "J":
                    t.junc = t.end
                    t.route = t.route[1:]
                    t.sec = t.route[0]
                    t.end, t.k = self.other_end(t.sec, t.junc), self.length[t.sec]
                else:
                    t.k -= 1
            # malfunction clock
            if t.malf > 0:
                t.malf -= 1
                if t.malf == 0:
                    events.append({"tick": now, "type": "malfunction_ended", "train": t.id})
            # arrival: one cell before the dead end of the destination
            if t.on and not t.junc and t.sec == t.dest and t.k == 1 and self.sec[t.sec]["kind"] == "terminus":
                t.arrived, t.on, t.blocked_by, t.route = now, False, None, []
                events.append({"tick": now, "type": "arrived", "train": t.id})
                continue
            if now == t.ed:
                events.append({"tick": now, "type": "departure_window_open", "train": t.id})
            was_blocked = t.blocked_by is not None
            if want[t.id] is not None and not moved and not stuck[t.id] and (pos[t.id] is not None or ready[t.id]):
                blocker = occupant.get(want[t.id])
                t.blocked_by = blocker if blocker and blocker != t.id else "contention"
                if not was_blocked:
                    events.append({"tick": now, "type": "blocked", "train": t.id, "by": t.blocked_by})
            else:
                t.blocked_by = None
            at_end = bool(t.on and not t.hold and len(t.route) == 1 and not t.junc and t.k == 0 and t.end != DEAD)
            if at_end and not t.at_end:
                events.append({"tick": now, "type": "reached_end_of_route", "train": t.id, "section": t.route[0]})
            t.at_end = at_end
        self.time = now
        for tid in self._deadlocked():
            if tid not in self.deadlocked:
                self.deadlocked.add(tid)
                events.append({"tick": now, "type": "deadlock", "train": tid})
        return events

    def _exits(self, t: _T) -> list:
        """Every cell the train could physically enter next, whatever its orders."""
        if t.junc:
            # It is entering t.sec. If t.sec has exactly one partner at this junction, the train came
            # from that partner (the stem) and could still take any of the stem's branches; otherwise
            # t.sec is the stem and is the only way out.
            partners = [x for x in self.sec if frozenset((x, t.sec)) in self.pairs[t.junc]]
            outs = [t.sec]
            if len(partners) == 1:
                outs = [x for x in self.sec if frozenset((x, partners[0])) in self.pairs[t.junc]]
            return [self._first_cell(x, t.junc) for x in outs]
        if t.k > 0:
            ends = self.sec[t.sec]["ends"]
            k = t.k - 1
            return [(t.sec, self.length[t.sec] - 1 - k if t.end == ends[1] else k)]
        if t.end == DEAD:
            return []
        return [("J", t.end)]

    def _deadlocked(self) -> list[str]:
        on = [t for t in self.trains if t.on and t.arrived is None]
        where = {self._cell(t): t.id for t in on}
        stuck = {t.id for t in on}
        exits = {t.id: self._exits(t) for t in on}
        changed = True
        while changed:
            changed = False
            for tid in list(stuck):
                nxt = exits[tid]
                if not nxt or any(where.get(c) not in stuck for c in nxt):
                    stuck.discard(tid)
                    changed = True
        return sorted(stuck)

    # ------------------------------------------------------------ interface
    @property
    def over(self) -> bool:
        return self.time >= self.horizon or all(t.arrived is not None for t in self.trains)

    def apply(self, commands: list[dict]) -> None:
        """Commands are assumed legal (the reference only issues legal ones)."""
        for c in commands:
            t = self.by_id[c["train"]]
            if c["action"] == "hold":
                t.hold = True
            else:
                t.route, t.hold, t.at_end = list(c["route"]), False, False

    def advance(self) -> None:
        self.events = []
        for _ in range(self.max_advance):
            if self.over:
                break
            ev = self.tick()
            self.events.extend(ev)
            if ev:
                break
        self.decision_id += 1

    def observation(self) -> dict:
        trains = []
        for t in self.trains:
            if t.arrived is not None:
                status = "arrived"
            elif t.malf > 0:
                status = "malfunction"
            elif not t.on:
                status = "ready" if self.time >= t.ed else "scheduled"
            else:
                status = "stopped" if (t.hold or t.at_end or t.blocked_by) else "running"
            loc = None
            if t.on:
                loc = {"section": t.sec, "heading_to": t.end, "cells_to_section_end": t.k}
                if t.junc:
                    loc = {"section": t.sec, "on_junction": t.junc, "heading_to": t.end, "cells_to_section_end": t.k}
            trains.append({"id": t.id, "status": status, "location": loc, "origin": t.origin, "destination": t.dest,
                           "earliest_departure": t.ed, "deadline": t.deadline, "route": list(t.route),
                           "holding": t.hold, "waiting_at_end_of_route": t.at_end, "blocked_by": t.blocked_by,
                           "malfunction_ticks_remaining": t.malf, "arrived_at": t.arrived})
        return {"decision_id": self.decision_id, "time": self.time, "horizon": self.horizon,
                "episode_over": self.over, "trains": trains, "events_since_last_decision": list(self.events)}

    def score(self) -> tuple[float, int, int]:
        """(score, completed, total lateness) under the published scoring rule."""
        total, done, late_sum = 0.0, 0, 0
        for t in self.trains:
            if t.arrived is not None:
                late = max(0, t.arrived - t.deadline)
                total += 1.0 - 0.5 * min(late, self.cap) / self.cap
                done += 1
                late_sum += late
        return total / len(self.trains), done, late_sum
