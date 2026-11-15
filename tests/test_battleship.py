"""Reward-integrity and rules tests for the Battleship duel (5x5 task board)."""
import json
import random
import threading

import pytest

from battleship.duel import (CELLS, Duel, DuelService, check_fleet, fleet_names, make_scenario, random_fleet,
                             replay, rules_text, uniform_fleet)
from battleship.duel_reference import ExactPlayer, all_layouts


def scenario(seed=101):
    return make_scenario(f"t{seed}", seed, "exact", uniform=True, disclose_opponent=True)


def random_game(seed, stop_after=None):
    """Random legal play through the service; returns (service, score after every accepted action)."""
    rng = random.Random(seed)
    svc = DuelService(scenario(100 + seed % 7))
    svc.place_fleet(fleet_names(random_fleet(rng)))
    cells = [f"{c}{r}" for c in "ABCDE" for r in range(1, 6)]
    rng.shuffle(cells)
    scores = [svc.game.result()["score"]]
    for i, cell in enumerate(cells):
        if svc.game.over or (stop_after is not None and i >= stop_after):
            break
        svc.fire(cell, f"r{i}")
        scores.append(svc.game.result()["score"])
    return svc, scores


def test_layout_count_and_cell_chances():
    a = all_layouts()
    assert len(a) == 5772
    share = lambda r, c: float(((a >> (r * 5 + c)) & 1).mean())  # noqa: E731
    assert round(share(0, 0), 3) == 0.608 and round(share(1, 1), 3) == 0.031 and round(share(2, 2), 3) == 0.523


def test_uniform_fleets_are_legal():
    rng = random.Random(3)
    for _ in range(200):
        check_fleet(fleet_names(uniform_fleet(rng)))


def test_opponent_is_disclosed_only_when_asked():
    assert "THE OPPONENT" in DuelService(scenario()).get_rules()["rules"]
    assert "THE OPPONENT" not in rules_text()


@pytest.mark.parametrize("seed", range(25))
def test_score_never_falls_and_stopping_never_beats_playing_on(seed):
    svc, scores = random_game(seed)
    assert all(b >= a for a, b in zip(scores, scores[1:])), scores
    assert 0.0 <= scores[-1] <= 1.0
    r = svc.game.result()
    if r["won"]:
        assert r["score"] > 0.5  # any win beats any loss or unfinished game (max 0.5 * 8/9)
    else:
        assert r["score"] <= 0.5 * (CELLS - 1) / CELLS


@pytest.mark.parametrize("seed", range(10))
def test_replay_of_the_log_gives_the_live_result(seed, tmp_path):
    from battleship.duel import load_actions, scenario_hash
    sc = scenario(100 + seed)
    svc = DuelService(sc, tmp_path / "e.jsonl")
    p = ExactPlayer()
    obs = svc.place_fleet(p.place())["observation"]
    n = 0
    while not obs["game_over"] and n < 3 + seed:
        obs = svc.fire(p.choose(obs), str(n))["observation"]
        n += 1
    header, actions = load_actions(tmp_path / "e.jsonl")
    assert header["scenario_hash"] == scenario_hash(sc)
    assert replay(sc, actions) == svc.game.result()


def test_do_nothing_and_place_only_score_zero():
    svc = DuelService(scenario())
    assert svc.game.result()["score"] == 0.0
    svc.place_fleet([["A1", "A2", "A3"], ["C1", "D1"], ["E3", "E4"], ["A5"], ["C5"]])
    assert svc.game.result()["score"] == 0.0


def test_illegal_requests_change_nothing():
    svc = DuelService(scenario())
    before = json.dumps(svc.observe())
    bad_fleets = [None, "[", [], [["A1"]], [["A1", "A2", "A3"], ["A4", "A5"], ["C1", "D1"], ["E3"], ["E5"]],  # touching
                  [["A1", "B2", "C3"], ["C1", "D1"], ["E3", "E4"], ["A5"], ["C5"]],  # diagonal ship
                  [["A1", "A2", "A3"], ["C1", "D1"], ["E3", "E4"], ["A5"], ["Z9"]],
                  [["A1", "A2", "A3", "A4"], ["C1", "D1"], ["E3", "E4"], ["C5"], ["E1"]]]
    for f in bad_fleets:
        assert svc.place_fleet(f)["status"] == "rejected"
    assert svc.fire("A1", "x")["status"] == "rejected"  # no fleet yet
    assert json.dumps(svc.observe()) == before and svc.actions == []
    assert svc.place_fleet([["A1", "A2", "A3"], ["C1", "D1"], ["E3", "E4"], ["A5"], ["C5"]])["status"] == "accepted"
    assert svc.place_fleet([["A1", "A2", "A3"], ["C1", "D1"], ["E3", "E4"], ["A5"], ["C5"]])["status"] == "rejected"
    first = svc.fire("C3", "one")
    assert first["status"] == "accepted"
    again = svc.fire("C3", "one")  # same request id: same answer, no second shot
    assert again["duplicate_request"] and svc.game.turn == 1
    for target, rid in (("C3", "two"), ("F1", "three"), ("A0", "four"), (None, "five"), ("A1", ""), ("A1", None)):
        assert svc.fire(target, rid)["status"] in ("rejected", "invalid_request")
    assert svc.game.turn == 1 and len(svc.actions) == 2


def test_observation_and_answers_never_reveal_unshot_enemy_cells():
    sc = scenario(105)
    svc = DuelService(sc)
    r = svc.place_fleet([["A1", "A2", "A3"], ["C1", "D1"], ["E3", "E4"], ["A5"], ["C5"]])
    r2 = svc.fire("C3", "a")
    for answer in (r, r2, svc.observe()):
        text = json.dumps(answer)
        assert "opponent_fleet" not in text and "score" not in text
        board = answer.get("observation", answer).get("opponent_board")
        if board:
            assert sum(ch in "x#" for row in board[1:] for ch in row[2:]) <= 1


def test_parallel_shots_at_one_cell_count_once():
    svc = DuelService(scenario())
    svc.place_fleet([["A1", "A2", "A3"], ["C1", "D1"], ["E3", "E4"], ["A5"], ["C5"]])
    out = []
    ts = [threading.Thread(target=lambda i=i: out.append(svc.fire("E5", f"p{i}")["status"])) for i in range(40)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert out.count("accepted") == 1 and svc.game.turn == 1


def test_shots_after_the_game_are_refused():
    svc = DuelService(scenario())
    p = ExactPlayer()
    obs = svc.place_fleet(p.place())["observation"]
    n = 0
    while not obs["game_over"]:
        obs = svc.fire(p.choose(obs), str(n))["observation"]
        n += 1
    final = svc.game.result()
    free = [f"{c}{r}" for c in "ABCDE" for r in range(1, 6)]
    assert all(svc.fire(x, "late" + x)["status"] == "rejected" for x in free)
    assert svc.game.result() == final


def test_opponent_shots_follow_from_public_information_only():
    """Rebuild every opponent shot from what an onlooker sees of the agent's board."""
    from battleship.duel import ExactOpponent, Side, parse
    sc = scenario(107)
    g = Duel(sc)
    mine = [["A1", "A2", "A3"], ["C1", "D1"], ["E3", "E4"], ["A5"], ["C5"]]
    g.place(mine)
    seen = []
    for cell in ("B2", "D4", "B4", "D2", "C3"):
        if g.over or cell in {x for x in g.enemy.shots}:
            continue
        seen += [s["cell"] for s in g.fire(cell)["opponent_shots"]]
    shadow, bot, rebuilt = Side(check_fleet(mine)), ExactOpponent(sc["opponent_shot_seed"]), []
    for cell in seen:
        guess = bot.choose(shadow)
        rebuilt.append(f"{'ABCDE'[guess[1]]}{guess[0] + 1}")
        shadow.receive(parse(cell))
    assert rebuilt == seen and len(seen) > 0
