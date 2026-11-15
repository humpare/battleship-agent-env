"""Local Battleship duel server for model-vs-algorithm games.

The opponent is the exact-elimination algorithm. Its fleet and seeds come from os.urandom and
exist only in this process's memory; nothing about the hidden state is written to disk until the
game is over. Usage: BATTLESHIP_N=5 python arena/server.py PORT ADMIN_KEY OUT.json [opponent]
"""
import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from battleship import duel
from battleship.duel import N, COLS, Duel, DuelService, fleet_names, parse, random_fleet
from battleship.duel_reference import ExactPlayer

port, key, out_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
kind = sys.argv[4] if len(sys.argv) > 4 else "exact"
secret = int.from_bytes(os.urandom(8), "big")


class ExactBot:
    """The exact algorithm as the opponent: it sees only what an attacker may see."""

    def __init__(self):
        self.p = ExactPlayer(seed=secret)

    def choose(self, target):
        return parse(self.p.choose({"opponent_board": target.public_rows(),
                                    "opponent_ships_afloat": target.afloat()}))


scenario = {"id": f"arena-{port}", "seed": 0,
            "opponent_fleet": fleet_names(random_fleet(random.Random(secret))),
            "opponent_shot_seed": secret % 10**9}
svc = DuelService(scenario)
if kind == "exact":
    bot = ExactBot()
    svc.game.bot = bot
    m = int(bot.p.all[random.Random(secret ^ 0xABCDEF).randrange(len(bot.p.all))])
    cells = [(r, c) for r in range(N) for c in range(N) if m >> (r * N + c) & 1]
    from battleship.duel_reference import groups
    svc.game.enemy = duel.Side([sorted(g) for g in groups(cells)])  # opponent fleet drawn uniformly from all legal layouts

RULES = duel.rules_text()


def finished():
    if svc.game.over and not os.path.exists(out_path):
        res = svc.game.result()
        res.update({"opponent": kind, "board": N, "actions": svc.actions,
                    "opponent_fleet": fleet_names(svc.game.enemy.ships)})
        json.dump(res, open(out_path, "w"), indent=1)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        cmd = self.path.strip("/")
        if cmd == "rules":
            ans = {"rules": RULES, "board_size": N, "fleet_lengths": duel.FLEET}
        elif cmd == "observe":
            ans = svc.observe()
        elif cmd == "place":
            ans = svc.place_fleet(body.get("ships"))
        elif cmd == "fire":
            ans = svc.fire(body.get("target"), body.get("request_id"))
            finished()
        elif cmd == "admin" and body.get("key") == key:
            ans = {**svc.game.result(), "actions": svc.actions, "opponent_fleet": fleet_names(svc.game.enemy.ships)}
        else:
            ans = {"status": "unknown_command"}
        data = json.dumps(ans).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


print("ready", port, flush=True)
HTTPServer(("127.0.0.1", port), H).serve_forever()
