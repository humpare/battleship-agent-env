"""Focused checks for the benchmark's concrete risks. Run: uv run pytest -q"""
import asyncio
import json
import socket
import subprocess
import sys
import time

import pytest

from dispatch.build import linear_line, make_scenario, train
from dispatch.reference import ReferenceDispatcher
from dispatch.rules import rules
from dispatch.runner import run_policy
from dispatch.service import EpisodeService, load_batches
from dispatch.sim import DispatchSim, replay

FULL_E = ["W_dep", "S1", "L1a", "S2", "E_arr"]
FULL_W = ["E_dep", "S2", "L1a", "S1", "W_arr"]


def two_trains(**kw):
    return {"id": "t2", "horizon": 100, "max_advance": 5, "lateness_cap": 30, "layout": linear_line(1),
            "trains": [train("T1", "E", 1, 60), train("T2", "W", 1, 60)], **kw}


def six():
    return make_scenario("six", loops=2, n_trains=6, seed=3, malfunctions=1)


def route(tid, r):
    return {"train": tid, "action": "route", "route": r}


# ---------------------------------------------------------------- determinism
def test_replay_reproduces_score_exactly():
    sc = six()
    res, log = run_policy(sc)
    again = replay(sc, log)
    assert again["per_train"] == res["per_train"] and again["score"] == res["score"]
    assert replay(sc, log) == again  # and replay itself is repeatable


def test_episodes_do_not_contaminate_each_other():
    sc = six()
    a, _ = run_policy(sc)
    run_policy(make_scenario("other", loops=1, n_trains=4, seed=9))
    DispatchSim(sc).apply([route("T1", ["W_dep", "S1"])])  # a half-used sim left lying around
    b, _ = run_policy(sc)
    assert a["per_train"] == b["per_train"]


def test_malfunction_schedule_is_independent_of_play():
    sc = two_trains(malfunctions=[{"tick": 10, "train": "T1", "duration": 5}])
    for commands in ([route("T1", FULL_E)], [route("T1", ["W_dep"])], []):
        sim = DispatchSim(sc)
        sim.apply(commands)
        while sim.time < 12:
            sim.tick()
        t1 = sim.observation()["trains"][0]
        assert t1["malfunction_ticks_remaining"] == 2, commands  # same event however T1 was driven


# ------------------------------------------------------------------ reference
def test_reference_completes_feasible_scenarios():
    for seed in range(25):
        sc = make_scenario(f"s{seed}", loops=1 + seed % 3, n_trains=4 + seed % 8, seed=seed,
                           headway=[0, 2, 4][seed % 3], malfunctions=seed % 3)
        res, _ = run_policy(sc)
        assert res["unfinished"] == 0 and not res["diagnostics"]["deadlocked_trains"], seed


def test_reference_uses_only_public_data():
    """The reference module must not import the simulator, and must run on JSON alone."""
    import dispatch.reference as ref_mod
    src = open(ref_mod.__file__).read()
    assert "flatland" not in src and "from ." not in src and "import dispatch" not in src
    sim = DispatchSim(six())
    net = json.loads(json.dumps(rules(sim)["network"]))
    obs = json.loads(json.dumps(sim.observation()))
    assert isinstance(ReferenceDispatcher(net).decide(obs), list)


# ------------------------------------------------------------ public / private
PRIVATE_WORDS = ("score", "reward", "total_lateness", "malfunctions", "seed", "duration")


def test_no_private_state_in_tool_answers():
    svc = EpisodeService(two_trains(malfunctions=[{"tick": 30, "train": "T2", "duration": 9}]))
    blobs = [json.dumps(svc.get_rules()["network"]), json.dumps(svc.get_rules()["timetable"]),
             json.dumps(svc.observe()),
             json.dumps(svc.step([route("T1", FULL_E)], 0, "a"))]
    for blob in blobs:
        for w in PRIVATE_WORDS:
            assert w not in blob.lower(), (w, blob[:200])
    # the future malfunction (tick 30, 9 ticks) is nowhere before it happens
    assert '"malfunction_ticks_remaining": 0' in blobs[2]


def test_observation_carries_what_the_rules_promise():
    sim = DispatchSim(six())
    obs = sim.observation()
    for k in ("schema", "episode_id", "decision_id", "time", "time_unit", "horizon", "ticks_remaining",
              "episode_over", "trains", "occupied_sections", "events_since_last_decision"):
        assert k in obs
    for k in ("id", "status", "location", "origin", "destination", "earliest_departure", "deadline",
              "route", "holding", "blocked_by", "malfunction_ticks_remaining", "arrived_at"):
        assert k in obs["trains"][0]
    net = rules(sim)["network"]
    assert {s["kind"] for s in net["sections"]} == {"origin", "terminus", "single", "loop"}


# ------------------------------------------------------------- step semantics
def test_duplicate_request_does_not_act_twice():
    svc = EpisodeService(two_trains())
    first = svc.step([route("T1", FULL_E)], 0, "req-1")
    t = svc.sim.time
    again = svc.step([route("T1", FULL_E)], 0, "req-1")
    assert again["duplicate_request"] and again["observation"] == first["observation"]
    assert svc.sim.time == t and len(svc.batches) == 1


def test_stale_decision_id_is_refused_without_advancing():
    svc = EpisodeService(two_trains())
    svc.step([], 0, "a")
    t = svc.sim.time
    ans = svc.step([], 0, "b")
    assert ans["status"] == "stale_decision" and not ans["time_advanced"] and svc.sim.time == t


def test_bad_batch_changes_nothing():
    svc = EpisodeService(two_trains())
    before = json.dumps(svc.observe())
    # first command is fine, second is not: nothing may be applied
    ans = svc.step([route("T1", FULL_E), route("T2", ["S2"])], 0, "a")
    assert ans["status"] == "rejected" and not ans["time_advanced"]
    assert json.dumps(svc.observe()) == before and svc.batches == []
    assert svc.step([{"train": "T1", "action": "fly"}], 0, "b")["status"] == "invalid_request"
    assert svc.step("not json", 0, "c")["status"] == "invalid_request"
    assert svc.step([], "0", "d")["status"] == "invalid_request"
    assert json.dumps(svc.observe()) == before


def test_illegal_routes_are_rejected():
    sim = DispatchSim(two_trains())
    for bad in (["S1"],  # does not start where the train is
                ["W_dep", "S1", "W_arr"],  # would need to reverse
                ["W_dep", "S1", "L1a", "S2", "E_dep"],  # may not enter an origin
                ["W_dep", "S1", "L1a", "L1b"],  # not connected
                ["W_dep", "nowhere"]):
        with pytest.raises(Exception):
            sim.validate([route("T1", bad)])
    assert sim.observation()["trains"][0]["route"] == []


def test_permitted_but_blocked_move_is_reported_not_rejected():
    """Head-on routes are accepted; occupancy then blocks them and the deadlock is reported."""
    svc = EpisodeService(two_trains())
    ans = svc.step([route("T1", FULL_E), route("T2", FULL_W)], 0, "a")
    assert ans["status"] == "accepted"
    events = []
    while not svc.sim.over:
        ans = svc.step([], svc.sim.decision_id, f"r{svc.sim.decision_id}")
        events += ans["observation"]["events_since_last_decision"]
    kinds = {e["type"] for e in events}
    assert "blocked" in kinds and "deadlock" in kinds
    res = svc.sim.result()
    assert res["completed"] == 0 and res["score"] == 0.0
    assert sorted(res["diagnostics"]["deadlocked_trains"]) == ["T1", "T2"]


def test_omitted_trains_keep_their_orders():
    sim = DispatchSim(two_trains())
    sim.apply([route("T1", ["W_dep", "S1", "L1a"])])
    for _ in range(6):
        sim.apply([])  # says nothing about T1
        sim.advance()
    t1 = sim.observation()["trains"][0]
    assert t1["location"]["section"] == "L1a" and t1["waiting_at_end_of_route"]


def test_hold_stops_a_train_until_rerouted():
    sim = DispatchSim(two_trains())
    sim.apply([route("T1", FULL_E)])
    while sim.time < 8:
        sim.tick()
    sim.apply([{"train": "T1", "action": "hold"}])
    where = sim.observation()["trains"][0]["location"]
    for _ in range(5):
        sim.tick()
    assert sim.observation()["trains"][0]["location"] == where


# ------------------------------------------------------------------- scoring
def test_do_nothing_and_hold_everything_score_zero():
    sc = six()
    assert replay(sc, [])["score"] == 0.0
    holds = [[{"train": t["id"], "action": "hold"} for t in sc["trains"]]]
    res = replay(sc, holds)
    assert res["score"] == 0.0 and res["unfinished"] == 6
    assert all(not p["departed"] for p in res["per_train"])  # never-departed trains stay in the books


def test_stopping_early_does_not_erase_obligations():
    sc = six()
    full, log = run_policy(sc)
    early, _ = run_policy(sc, max_decisions=6)
    assert early["trains"] == full["trains"] == 6
    assert early["unfinished"] > 0 and early["score"] < full["score"]
    assert early["final_tick"] == sc["horizon"]  # the clock still ran out on the unfinished trains


def test_completion_requires_reaching_the_destination():
    """A train parked one section short of its terminus is not complete."""
    sim = DispatchSim(two_trains())
    sim.apply([route("T1", ["W_dep", "S1", "L1a", "S2"])])
    sim.run_out()
    p = sim.result()["per_train"][0]
    assert p["departed"] and not p["completed"] and p["score"] == 0.0


def test_late_arrival_beats_no_arrival_and_on_time_beats_late():
    sc = two_trains()
    sc["trains"][0]["deadline"] = 20  # T1 cannot make this
    res = replay(sc, [[route("T1", FULL_E)]])
    p = res["per_train"][0]
    assert p["completed"] and p["lateness"] > 0 and 0.5 <= p["score"] < 1.0
    assert res["per_train"][1]["score"] == 0.0  # T2 never ran


def test_step_rewards_sum_to_final_score(tmp_path):
    sc = six()
    log = tmp_path / "ep.jsonl"
    svc = EpisodeService(sc, log)
    ref = ReferenceDispatcher(svc.get_rules()["network"])
    obs = svc.observe()
    while not obs["episode_over"]:
        obs = svc.step(ref.decide(obs), obs["decision_id"], f"r{obs['decision_id']}")["observation"]
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    total = sum(r.get("score_delta", 0) for r in recs)
    header, batches = load_batches(log)
    final = replay(sc, batches)["score"]
    assert abs(total - final) < 1e-6 and final == svc.sim.result()["score"]


def test_counterfactual_branch_is_reproducible():
    """Restore a mid-episode state by replay, then compare two continuations."""
    sc = two_trains()
    prefix = [[route("T1", ["W_dep", "S1", "L1a"]), route("T2", ["E_dep", "S2"])]] + [[]] * 3

    def branch(decision):
        sim = DispatchSim(sc)
        for c in prefix:
            sim.apply(c)
            sim.advance()
        sim.apply(decision)
        ref = ReferenceDispatcher(rules(sim)["network"])
        while not sim.over:  # same continuation policy on both branches
            sim.advance()
            if not sim.over:
                sim.apply(ref.decide(sim.observation()))
        return sim.result()

    bad = branch([route("T2", ["S2", "L1a"])])  # into the track T1 is standing on
    good = branch([route("T2", ["S2", "L1b"])])
    assert good["completed"] == 2 and bad["completed"] == 0
    assert branch([route("T2", ["S2", "L1a"])]) == bad


# ----------------------------------------------------------------------- MCP
def test_real_mcp_client_completes_an_episode(tmp_path):
    from dispatch.mcp_client import play
    sc = six()
    (tmp_path / "sc.json").write_text(json.dumps(sc))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-W", "ignore", "-m", "dispatch.server", "--scenario", str(tmp_path / "sc.json"),
         "--log", str(tmp_path / "ep.jsonl"), "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        out = asyncio.run(play(f"http://127.0.0.1:{port}/mcp"))
    finally:
        proc.terminate()
    assert out["tools"] == ["get_rules", "observe", "step"]
    assert out["arrived"] == out["trains"] == 6
    _, batches = load_batches(tmp_path / "ep.jsonl")
    assert replay(sc, batches)["completed"] == 6


# ------------------------------------------------------------ model runners
def test_stateless_runner_scores_through_the_same_replay(tmp_path):
    """A stand-in 'model' that answers with the reference's commands must get the reference's score."""
    from dispatch.llm_runner import parse_commands, run, slim_observation
    sc = six()
    sim = DispatchSim(sc)
    ref = ReferenceDispatcher(rules(sim)["network"])
    shadow = EpisodeService(sc)  # the stand-in tracks the true observation itself

    def chat(messages, tools=None):
        obs = shadow.observe()
        assert json.loads(messages[-1]["content"]) == json.loads(json.dumps(slim_observation(obs)))
        cmds = ref.decide(obs)
        shadow.step(cmds, obs["decision_id"], f"x{obs['decision_id']}")
        return {"role": "assistant", "content": "<think>plan</think>" + json.dumps(cmds)}

    res = run(sc, chat, "stateless", tmp_path / "r")
    expect, _ = run_policy(sc)
    assert res["score"] == expect["score"] and res["refused_batches"] == 0
    assert parse_commands('noise [{"train":"T1","action":"hold"}] tail') == [{"train": "T1", "action": "hold"}]
    assert parse_commands("no list here") is None


def test_stateless_runner_gives_no_credit_for_garbage(tmp_path):
    from dispatch.llm_runner import run
    res = run(six(), lambda m, tools=None: {"role": "assistant", "content": "I think we should go."},
              "stateless", tmp_path / "g")
    assert res["score"] == 0.0 and res["unparsable_replies"] == res["steps"] > 0


def test_random_play_never_crashes_and_always_replays():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("fuzz", pathlib.Path(__file__).with_name("fuzz_random_play.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(25)
