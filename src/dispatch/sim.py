"""Dispatching simulator: Flatland 4.3.0 behind a named-graph command interface.

Controllers never touch Flatland directly. They issue two kinds of standing
orders per train:

* ``route``: an ordered list of sections, starting with the section the train
  is in. The train runs through them and stops at the far end of the last one.
* ``hold``: stop where you are until a new route is given.

Each tick the simulator turns standing orders into native Flatland actions
(MOVE_LEFT / MOVE_FORWARD / MOVE_RIGHT / STOP_MOVING / DO_NOTHING). It does not
check for conflicts between trains: Flatland's occupancy rules decide what
physically happens.
"""
from __future__ import annotations

import copy
import hashlib
import json
import warnings
from dataclasses import dataclass, field

from flatland.core.effects_generator import EffectsGenerator
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions as A
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState
from flatland.envs.timetable_utils import Line, Timetable

from .layout import DELTA, Layout

OBS_SCHEMA = "dispatch-obs/1"
MOVES = (A.MOVE_FORWARD, A.MOVE_LEFT, A.MOVE_RIGHT)


class CommandError(Exception):
    """A command batch that is well-formed but not allowed by the rules."""


class RequestError(Exception):
    """A command batch that is not well-formed."""


@dataclass
class Train:
    id: str
    handle: int
    origin: str
    destination: str
    earliest_departure: int
    deadline: int
    route: list[str] = field(default_factory=list)  # route[0] = current / entering section
    hold: bool = False
    arrival_tick: int | None = None
    blocked_by: str | None = None
    blocked_ticks: int = 0
    at_route_end: bool = False
    malfunction_left: int = 0


class _ScheduledMalfunctions(EffectsGenerator):
    """Exogenous events: fixed (tick, train, duration) triples, independent of play."""

    def __init__(self, schedule: list[tuple[int, int, int]]):
        super().__init__()
        self.schedule = schedule

    def on_episode_start(self, env, *args, **kwargs):
        return env

    def on_episode_step_start(self, env, *args, **kwargs):
        for tick, handle, duration in self.schedule:
            if tick == env._elapsed_steps:
                env.agents[handle].malfunction_handler.malfunction_down_counter = duration
        return env

    def on_episode_step_end(self, env, *args, **kwargs):
        return env


class DispatchSim:
    def __init__(self, scenario: dict):
        self.scenario = copy.deepcopy(scenario)
        self.id = scenario["id"]
        self.horizon = int(scenario["horizon"])
        self.max_advance = int(scenario.get("max_advance", 5))
        # Lateness at which a completed train's credit bottoms out at 0.5 (one free-run journey time).
        self.lateness_cap = int(scenario["lateness_cap"])
        if self.lateness_cap < 1:
            raise ValueError("lateness_cap must be at least 1 tick")
        self.layout = Layout.from_spec(scenario["layout"])
        self.trains: list[Train] = []
        for h, t in enumerate(scenario["trains"]):
            tr = Train(t["id"], h, t["origin"], t["destination"], int(t["earliest_departure"]), int(t["deadline"]))
            if self.layout.sections[tr.origin].kind != "origin":
                raise ValueError(f"{tr.id}: origin {tr.origin} is not an origin section")
            if self.layout.sections[tr.destination].kind != "terminus":
                raise ValueError(f"{tr.id}: destination {tr.destination} is not a terminus section")
            if not (1 <= tr.earliest_departure < tr.deadline < self.horizon):
                raise ValueError(f"{tr.id}: need 1 <= earliest_departure < deadline < horizon")
            self.trains.append(tr)
        self.by_id = {t.id: t for t in self.trains}
        if len(self.by_id) != len(self.trains):
            raise ValueError("duplicate train ids")
        sched = [(int(m["tick"]), self.by_id[m["train"]].handle, int(m["duration"]))
                 for m in scenario.get("malfunctions", [])]
        self.env = self._build_env(sched)
        self.rail = self.env.rail
        self.decision_id = 0
        self.events: list[dict] = []
        self.diag = {"blocked_train_ticks": 0, "deadlocked_trains": [], "malfunctions_seen": 0}

    # ------------------------------------------------------------ flatland
    def _build_env(self, sched) -> RailEnv:
        lay = self.layout
        rail = lay.to_rail()

        def rail_gen(*_a, **_k):
            return rail, {}

        def line_gen(*_a, **_k):
            wps = {}
            for tr in self.trains:
                pos, d = lay.start_config(tr.origin)
                wps[tr.handle] = [[Waypoint(pos, d)], [Waypoint(lay.target_cell(tr.destination), None)]]
            return Line(agent_waypoints=wps, agent_speeds=[1.0] * len(self.trains))

        def tt_gen(*_a, **_k):
            return Timetable(
                earliest_departures=[[t.earliest_departure, None] for t in self.trains],
                latest_arrivals=[[None, t.deadline] for t in self.trains],
                max_episode_steps=self.horizon,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = RailEnv(
                width=lay.width, height=lay.height,
                rail_generator=rail_gen, line_generator=line_gen, timetable_generator=tt_gen,
                number_of_agents=len(self.trains),
                obs_builder_object=DummyObservationBuilder(),
                effects_generator=_ScheduledMalfunctions(sched),
                remove_agents_at_target=True, random_seed=0,
            )
            env.reset(random_seed=0)
        return env

    @property
    def time(self) -> int:
        return self.env._elapsed_steps

    @property
    def over(self) -> bool:
        return self.time >= self.horizon or all(t.arrival_tick is not None for t in self.trains)

    def _agent(self, tr: Train):
        return self.env.agents[tr.handle]

    def _config(self, tr: Train):
        return self._agent(tr).current_configuration if self._agent(tr).state.is_on_map_state() else None

    def _successor(self, cfg):
        """Next cell straight along plain track (None on a junction cell)."""
        pos, d = cfg
        trans = self.rail.get_transitions((pos, d))
        outs = [k for k in range(4) if trans[k]]
        if len(outs) != 1:
            return None
        return (pos[0] + DELTA[outs[0]][0], pos[1] + DELTA[outs[0]][1])

    def _heading_end(self, tr: Train) -> str | None:
        """Junction (or None = dead end) the train will reach at the end of route[0]."""
        sec = self.layout.sections[tr.route[0]] if tr.route else self.layout.sections[tr.origin]
        cfg = self._config(tr)
        if cfg is None:  # not yet on the network: leaves the origin stub by its junction end
            return sec.a if sec.a is not None else sec.b
        pos, _ = cfg
        if pos in self.layout.junction_at:
            return sec.other_end(self.layout.junction_at[pos])
        nxt = self._successor(cfg)
        _, idx = self.layout.cell_index[pos]
        if nxt in self.layout.junction_at:
            return self.layout.junction_at[nxt]
        if nxt in self.layout.cell_index:
            return sec.b if self.layout.cell_index[nxt][1] > idx else sec.a
        return None

    # ------------------------------------------------------------ commands
    def validate(self, commands) -> list[tuple[Train, str, list[str]]]:
        """Check a whole batch without changing anything. Raises on the first problem."""
        if not isinstance(commands, list):
            raise RequestError("commands must be a list")
        parsed, seen = [], set()
        for i, c in enumerate(commands):
            if not isinstance(c, dict):
                raise RequestError(f"commands[{i}] must be an object")
            extra = set(c) - {"train", "action", "route"}
            if extra:
                raise RequestError(f"commands[{i}] has unknown fields {sorted(extra)}")
            tid, action = c.get("train"), c.get("action")
            if not isinstance(tid, str) or action not in ("route", "hold"):
                raise RequestError(f"commands[{i}] needs a string 'train' and 'action' of 'route' or 'hold'")
            route = c.get("route", [])
            if action == "route" and (not isinstance(route, list) or not route
                                      or not all(isinstance(s, str) for s in route)):
                raise RequestError(f"commands[{i}]: 'route' must be a non-empty list of section names")
            if action == "hold" and "route" in c:
                raise RequestError(f"commands[{i}]: 'hold' takes no route")
            if tid not in self.by_id:
                raise CommandError(f"commands[{i}]: unknown train {tid!r}")
            if tid in seen:
                raise CommandError(f"commands[{i}]: train {tid} appears twice in one batch")
            seen.add(tid)
            tr = self.by_id[tid]
            if tr.arrival_tick is not None:
                raise CommandError(f"commands[{i}]: train {tid} has already arrived")
            if action == "route":
                self._check_route(i, tr, route)
            parsed.append((tr, action, list(route)))
        return parsed

    def _check_route(self, i: int, tr: Train, route: list[str]) -> None:
        lay = self.layout
        for s in route:
            if s not in lay.sections:
                raise CommandError(f"commands[{i}]: unknown section {s!r}")
        if len(set(route)) != len(route):
            raise CommandError(f"commands[{i}]: a route may not repeat a section")
        current = tr.route[0] if tr.route else tr.origin
        if route[0] != current:
            raise CommandError(f"commands[{i}]: route for {tr.id} must start with its current section {current!r}")
        end = self._heading_end(tr)
        for prev, nxt in zip(route, route[1:]):
            if end is None:
                raise CommandError(f"commands[{i}]: {prev} ends in a dead end in the train's direction of travel")
            j = lay.junctions[end]
            if nxt not in (j.stem, *j.branches) or not j.connected(prev, nxt):
                raise CommandError(
                    f"commands[{i}]: {tr.id} leaves {prev} through {end}, which does not lead to {nxt} "
                    f"(trains cannot reverse)")
            kind = lay.sections[nxt].kind
            if kind == "origin":
                raise CommandError(f"commands[{i}]: trains may not enter origin section {nxt}")
            if kind == "terminus" and nxt != tr.destination:
                raise CommandError(f"commands[{i}]: {nxt} is not the destination of {tr.id}")
            end = lay.sections[nxt].other_end(end)

    def apply(self, commands) -> None:
        parsed = self.validate(commands)
        for tr, action, route in parsed:
            if action == "hold":
                tr.hold = True
            else:
                tr.route, tr.hold, tr.at_route_end = route, False, False

    # ---------------------------------------------------------------- tick
    def _plan(self, tr: Train):
        """Native action for this tick and the cell the train is trying to enter."""
        ag = self._agent(tr)
        st = ag.state
        if st == TrainState.DONE:
            return A.DO_NOTHING, None
        if st.is_off_map_state():
            if tr.hold or not tr.route or self.time + 1 < tr.earliest_departure:
                return A.DO_NOTHING, None
            return A.MOVE_FORWARD, ag.initial_configuration[0]
        cfg = ag.current_configuration
        if tr.hold:
            return A.STOP_MOVING, None
        pos, _ = cfg
        if pos in self.layout.junction_at:
            j = self.layout.junctions[self.layout.junction_at[pos]]
            want = self.layout._adjacent_cell(j, tr.route[0])
        else:
            want = self._successor(cfg)
            if want in self.layout.junction_at:
                if len(tr.route) < 2:
                    return A.STOP_MOVING, None  # end of authority: wait before the junction
            elif want not in self.layout.cell_index:
                return A.STOP_MOVING, None
        for a in MOVES:
            res = self.rail.apply_action_independent(a, cfg)
            if res is not None and res[0][0] == want:
                return a, want
        raise RuntimeError(f"no native action takes {tr.id} from {cfg} to {want}")

    def tick(self) -> list[dict]:
        """Advance exactly one Flatland step. Returns the events of this tick."""
        if self.over:
            raise RuntimeError("episode is over")
        events: list[dict] = []
        before = {tr.id: self._config(tr) for tr in self.trains}
        occupant = {cfg[0]: tid for tid, cfg in before.items() if cfg is not None}
        could_enter = {tr.id: self._agent(tr).state in (TrainState.READY_TO_DEPART, TrainState.MALFUNCTION_OFF_MAP)
                       for tr in self.trains}
        plans = {tr.id: self._plan(tr) for tr in self.trains}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.env.step({tr.handle: plans[tr.id][0] for tr in self.trains})
        now = self.time
        for tr in self.trains:
            if tr.arrival_tick is not None:
                continue
            ag = self._agent(tr)
            action, want = plans[tr.id]
            # malfunction bookkeeping (counter is read after Flatland's own decrement)
            left = ag.malfunction_handler.malfunction_down_counter
            in_malf = ag.state.is_malfunction_state() or left > 0
            if left > 0 and tr.malfunction_left == 0:
                events.append({"tick": now, "type": "malfunction_started", "train": tr.id,
                               "ticks_remaining": left})
                self.diag["malfunctions_seen"] += 1
            if left == 0 and tr.malfunction_left > 0:
                events.append({"tick": now, "type": "malfunction_ended", "train": tr.id})
            tr.malfunction_left = left  # the train will not move during the next `left` ticks

            if ag.state == TrainState.DONE:
                tr.arrival_tick = now
                tr.blocked_by, tr.route = None, []
                events.append({"tick": now, "type": "arrived", "train": tr.id})
                continue
            if now == tr.earliest_departure:
                events.append({"tick": now, "type": "departure_window_open", "train": tr.id})
            cfg = self._config(tr)
            if cfg is not None and cfg[0] in self.layout.junction_at and before[tr.id] is not None \
                    and before[tr.id][0] != cfg[0]:
                tr.route = tr.route[1:]  # left route[0]; now entering the next section
            # blocked: tried to move, was not broken down, did not get the cell
            moved = cfg is not None and (before[tr.id] is None or before[tr.id][0] != cfg[0])
            was_blocked = tr.blocked_by is not None
            if want is not None and not moved and not in_malf and (before[tr.id] is not None or could_enter[tr.id]):
                blocker = occupant.get(want)
                tr.blocked_by = blocker if blocker and blocker != tr.id else "contention"
                tr.blocked_ticks += 1
                self.diag["blocked_train_ticks"] += 1
                if not was_blocked:
                    events.append({"tick": now, "type": "blocked", "train": tr.id, "by": tr.blocked_by})
            else:
                tr.blocked_by = None
            # reached the end of its authority
            at_end = False
            if cfg is not None and not tr.hold and len(tr.route) == 1 and cfg[0] not in self.layout.junction_at:
                at_end = self._successor(cfg) in self.layout.junction_at
            if at_end and not tr.at_route_end:
                events.append({"tick": now, "type": "reached_end_of_route", "train": tr.id,
                               "section": tr.route[0]})
            tr.at_route_end = at_end
        for tid in self._deadlocked():
            if tid not in self.diag["deadlocked_trains"]:
                self.diag["deadlocked_trains"].append(tid)
                events.append({"tick": now, "type": "deadlock", "train": tid})
        return events

    def _deadlocked(self) -> list[str]:
        """Trains that can never move again: every cell they could enter is held
        by another train in the same stuck set. Trains cannot reverse, so this is permanent."""
        cfgs = {tr.id: self._config(tr) for tr in self.trains}
        stuck = {tid for tid, c in cfgs.items() if c is not None}
        where = {c[0]: tid for tid, c in cfgs.items() if c is not None}
        changed = True
        while changed:
            changed = False
            for tid in list(stuck):
                pos, d = cfgs[tid]
                trans = self.rail.get_transitions((pos, d))
                nxt = [(pos[0] + DELTA[k][0], pos[1] + DELTA[k][1]) for k in range(4) if trans[k]]
                if not nxt or any(where.get(c) not in stuck for c in nxt):
                    stuck.discard(tid)
                    changed = True
        return sorted(stuck)

    def advance(self) -> None:
        """Run until something happens that a dispatcher may want to react to:
        at least 1 tick, at most ``max_advance`` ticks."""
        self.events = []
        for _ in range(self.max_advance):
            if self.over:
                break
            ev = self.tick()
            self.events.extend(ev)
            if ev:
                break
        self.decision_id += 1

    def run_out(self) -> None:
        """No more commands will come: standing orders run to the horizon."""
        while not self.over:
            self.tick()

    # ---------------------------------------------------------- public view
    def observation(self) -> dict:
        lay = self.layout
        trains, occupancy = [], {}
        for tr in self.trains:
            ag = self._agent(tr)
            cfg = self._config(tr)
            if tr.arrival_tick is not None:
                status = "arrived"
            elif tr.malfunction_left > 0:
                status = "malfunction"
            elif cfg is None:
                status = "ready" if self.time >= tr.earliest_departure else "scheduled"
            else:
                status = "stopped" if (tr.hold or tr.at_route_end or tr.blocked_by) else "running"
            loc = None
            if cfg is not None:
                pos, _ = cfg
                end = self._heading_end(tr)
                if pos in lay.junction_at:
                    loc = {"section": tr.route[0], "on_junction": lay.junction_at[pos],
                           "heading_to": end or "dead_end",
                           "cells_to_section_end": lay.sections[tr.route[0]].length}
                else:
                    sec = lay.sections[tr.route[0]]
                    idx = lay.cell_index[pos][1]
                    to_end = (sec.length - 1 - idx) if end == sec.b else idx
                    loc = {"section": sec.name, "heading_to": end or "dead_end", "cells_to_section_end": to_end}
                occupancy.setdefault(tr.route[0], []).append(tr.id)
            trains.append({
                "id": tr.id, "status": status, "location": loc,
                "origin": tr.origin, "destination": tr.destination,
                "earliest_departure": tr.earliest_departure, "deadline": tr.deadline,
                "route": list(tr.route), "holding": tr.hold,
                "waiting_at_end_of_route": tr.at_route_end,
                "blocked_by": tr.blocked_by,
                "malfunction_ticks_remaining": tr.malfunction_left,
                "arrived_at": tr.arrival_tick,
            })
        return {
            "schema": OBS_SCHEMA, "episode_id": self.id, "decision_id": self.decision_id,
            "time": self.time, "time_unit": "tick", "horizon": self.horizon,
            "ticks_remaining": max(0, self.horizon - self.time), "episode_over": self.over,
            "trains": trains, "occupied_sections": occupancy,
            "events_since_last_decision": list(self.events),
        }

    # -------------------------------------------------------------- scoring
    def result(self) -> dict:
        """Authoritative outcome. Never sent to a controller."""
        per = []
        for tr in self.trains:
            done = tr.arrival_tick is not None
            late = max(0, tr.arrival_tick - tr.deadline) if done else None
            cap = self.lateness_cap
            score = (1.0 - 0.5 * min(late, cap) / cap) if done else 0.0
            per.append({"train": tr.id, "completed": done, "arrival_tick": tr.arrival_tick,
                        "deadline": tr.deadline, "lateness": late, "score": round(score, 6),
                        "departed": done or self._config(tr) is not None,
                        "blocked_ticks": tr.blocked_ticks})
        n = len(per)
        return {
            "scenario": self.id, "scenario_hash": scenario_hash(self.scenario),
            "score": round(sum(p["score"] for p in per) / n, 6),
            "completed": sum(p["completed"] for p in per), "trains": n,
            "unfinished": n - sum(p["completed"] for p in per),
            "total_lateness": sum(p["lateness"] or 0 for p in per),
            "final_tick": self.time, "decisions": self.decision_id,
            "diagnostics": self.diag, "per_train": per,
        }


def scenario_hash(scenario: dict) -> str:
    return hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()[:16]


def replay(scenario: dict, batches: list[list[dict]]) -> dict:
    """Re-run an action log from scratch and score it. Used by the verifier."""
    sim = DispatchSim(scenario)
    for commands in batches:
        if sim.over:
            break
        sim.apply(commands)
        sim.advance()
    sim.run_out()
    return sim.result()
