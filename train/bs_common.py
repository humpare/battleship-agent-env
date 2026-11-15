"""Shared by the small-model experiment: the prompt a small model sees for each Battleship
decision, and how its answer is read. One decision = one short chat turn (no tool calling),
so a 3B model can be tested before and after training through exactly the same interface."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from battleship.duel import COLS, FLEET, N, make_scenario  # noqa: E402

_lengths = sorted(FLEET, reverse=True)
SYSTEM = f"""You are playing Battleship on a {N}x{N} board (columns A-{COLS[-1]}, rows 1-{N}) against a program.
Each side hides a fleet of {len(FLEET)} ships with lengths {", ".join(map(str, _lengths))}. Ships are straight (horizontal or vertical) and never touch each other, not even diagonally.
You shoot one cell at a time. A hit or a sinking gives you another shot; a miss passes the turn to the opponent. The first side to sink the whole enemy fleet wins.
The opponent's fleet was drawn uniformly at random from all legal fleets. The opponent shoots the cell most likely to hold one of your ships, given the results of its shots so far.
Board symbols: . unknown, o miss, x hit on a ship not yet sunk, # sunk ship.
Answer with the move only."""

PLACE = ("Place your fleet. Answer with the ships as JSON, each ship a list of its cells, "
         'for example [["A1","A2","A3"],["C1","D1"],["E3","E4"],["A5"],["C5"]].')


def shot_prompt(obs: dict) -> str:
    return ("Opponent board:\n" + "\n".join(obs["opponent_board"]) +
            "\nOpponent ships still afloat (lengths): " + ", ".join(map(str, obs["opponent_ships_afloat"])) +
            "\nYour shot (one cell):")


def messages(user: str, answer: str | None = None) -> list[dict]:
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return m + [{"role": "assistant", "content": answer}] if answer is not None else m


def read_shot(text: str) -> str | None:
    m = re.search(rf"\b([A-{COLS[-1]}])\s?([1-{N}])\b", text.upper())
    return m.group(1) + m.group(2) if m else None


def read_fleet(text: str):
    for t in (text, "[" + text, text + "]"):  # tolerate one missing outer bracket
        m = re.search(r"\[\s*\[.*\]\s*\]", t, re.S)
        try:
            if m:
                return json.loads(m.group(0))
        except ValueError:
            pass
    return None


def eval_scenarios() -> dict[str, list[dict]]:
    """Boards never used for training: the 30 frozen task boards and 100 more."""
    mk = lambda s: make_scenario(f"bs{N}-s{s}", s, "exact", uniform=True, disclose_opponent=True)  # noqa: E731
    return {"frozen30": [mk(s) for s in range(101, 131)], "extra100": [mk(s) for s in range(1001, 1101)]}
