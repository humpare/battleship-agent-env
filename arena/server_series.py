"""A series of Battleship games against the same opponent. Both sides place their own fleet in
every game. The opponent is the exact search. Across the series it learns in three ways: it places where
the player's past shots come last, it treats its own revealed fleets as known to the player and
never repeats one, and it weights its search by where the player has placed ships before.
Everything is determined by the series seed and the moves played: no hidden randomness.

Usage: BATTLESHIP_N=5 python arena/server_series.py PORT ADMIN_KEY SERIES_SEED [GAMES]
"""
import json
import random
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from battleship import duel
from battleship.adaptive import AdaptivePlacer
from battleship.duel import N, DuelService, fleet_names, name
from battleship.duel_reference import ExactPlayer

port, key, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
GAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 5
base = ExactPlayer()
duel.ExactOpponent._shared = base
table = Path(duel.__file__).with_name(f"placement_{N}{'w' if duel.WRAP else ''}.json")
TABLE = json.loads(table.read_text())["best"] if table.exists() else None
placer = AdaptivePlacer(base.all, seed=seed, table=TABLE, pick_from=12)

SERIES_RULES = f"""\
THE SERIES
- You play {GAMES} games in a row against the SAME opponent. Your result is the average over all {GAMES} games.
- In every game BOTH sides place their own fleet: you choose yours with `place`, the opponent
  chooses its own. You may place differently in each game.
- The opponent does not change between games, and it may adapt to how you played earlier
  (for example, where you tend to shoot). You may adapt to it in the same way.
- After a game ends, `history` shows every finished game in full: both fleets (the opponent's
  is revealed once the game is over) and every shot by both sides in order.
- When a game is over, run `next` to start the following game. You shoot first in every game.

"""
RULES = duel.rules_text().replace("SETUP\n", SERIES_RULES + "SETUP\n").replace(
    "There is one game and no reset.", "")

state = {"game": 0, "svc": None, "finished": [], "player_shot_logs": []}


def start_game():
    state["game"] += 1
    g = state["game"]
    past = state["finished"]
    # placement: learns where the player shoots, treats its own revealed fleets as burned, never repeats one
    opp_fleet = placer.place(state["player_shot_logs"], [f["opponent_fleet"] for f in past])
    sc = {"id": f"series-{seed}-g{g}", "seed": g, "opponent": "exact", "opponent_fleet": opp_fleet,
          "opponent_shot_seed": seed * 100 + g}
    state["svc"] = DuelService(sc)
    # search: weights the possible fleets by where this player has placed ships before
    state["svc"].game.bot.player.learn_placement([f["your_fleet"] for f in past])
    state["placer_note"] = dict(placer.last)


def close_game():
    svc = state["svc"]
    g = svc.game
    mine, enemy = g.mine, g.enemy
    player_shots = [{"cell": c, "result": r} for c, r in _ordered(enemy)]
    opp_shots = [{"cell": c, "result": r} for c, r in _ordered(mine)]
    state["player_shot_logs"].append(player_shots)
    res = g.result()
    state["finished"].append({
        "game": state["game"], "winner": g.winner, "score": res["score"], "margin": res["margin"],
        "your_fleet": fleet_names(mine.ships), "opponent_fleet": fleet_names(enemy.ships),
        "your_shots": [f"{name(*s['cell'])}:{s['result']}" for s in player_shots],
        "opponent_shots": [f"{name(*s['cell'])}:{s['result']}" for s in opp_shots],
        "placer": state["placer_note"]})


def _ordered(side):
    """Shots received by a side, in order, with 'sunk' marked on the shot that finished a ship."""
    seen, out = set(), []
    for cell, res in side.shots.items():
        seen.add(cell)
        if res == "hit":
            ship = next(s for s in side.ships if cell in s)
            res = "sunk" if all(x in seen for x in ship) else "hit"
        out.append((cell, res))
    return out


def public_history():
    return [{k: v for k, v in f.items() if k not in ("score", "margin", "placer")} for f in state["finished"]]


start_game()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        cmd, svc = self.path.strip("/"), state["svc"]
        closed = len(state["finished"]) == state["game"]
        if cmd == "rules":
            ans = {"rules": RULES}
        elif cmd == "observe":
            ans = svc.observe()
        elif cmd == "place":
            ans = svc.place_fleet(body.get("ships"))
        elif cmd == "fire":
            ans = svc.fire(body.get("target"), body.get("request_id"))
            if svc.game.over and not closed:
                close_game()
        elif cmd == "history":
            ans = {"games_finished": len(state["finished"]), "games_total": GAMES, "history": public_history()}
        elif cmd == "next":
            if not svc.game.over:
                ans = {"status": "rejected", "message": "the current game is not over yet"}
            elif state["game"] >= GAMES:
                ans = {"status": "series_over", "message": f"all {GAMES} games have been played"}
            else:
                start_game()
                ans = {"status": "accepted", "message": f"game {state['game']} of {GAMES} started: place your fleet",
                       "observation": state["svc"].observe()}
        elif cmd == "admin" and body.get("key") == key:
            ans = {"games": state["finished"], "current_game": state["game"]}
        else:
            ans = {"status": "unknown_command"}
        if isinstance(ans, dict):
            ans.setdefault("series", f"game {state['game']} of {GAMES}")
        data = json.dumps(ans).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


print("ready", port, "series", seed, flush=True)
HTTPServer(("127.0.0.1", port), H).serve_forever()
