"""Search-only duel: the server deals BOTH fleets at random; the player only shoots.
Deals depend only on the game number, so every player gets the same pair of fleets in game k.
The opponent is the exact algorithm with a fixed random tie-break (no favourite cells).
Usage: BATTLESHIP_N=6 BATTLESHIP_WRAP=1 python arena/server_dealt.py PORT ADMIN_KEY GAME_NUMBER
"""
import json
import random
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from battleship import duel
from battleship.duel import N, DuelService, fleet_names, parse
from battleship.duel_reference import ExactPlayer, groups

port, key, game = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])


def deal(player: ExactPlayer, seed: int):
    m = int(player.all[random.Random(seed).randrange(len(player.all))])
    cells = [(r, c) for r in range(N) for c in range(N) if m >> (r * N + c) & 1]
    return [sorted(g) for g in groups(cells)]


class Bot:
    def __init__(self):
        self.p = ExactPlayer(tie_seed=1000 + game)

    def choose(self, target):
        return parse(self.p.choose({"opponent_board": target.public_rows(), "opponent_ships_afloat": target.afloat()}))


bot = Bot()
enemy, mine = deal(bot.p, 7000 + game), deal(bot.p, 9000 + game)
svc = DuelService({"id": f"dealt-{game}", "seed": game, "opponent_fleet": fleet_names(enemy), "opponent_shot_seed": game})
svc.game.bot = bot
svc.game.place(fleet_names(mine))
RULES = (duel.rules_text()
         .replace("SETUP\n- place_fleet(ships): place your own fleet before the first shot.",
                  "SETUP\n- Your fleet has ALREADY been placed for you at random, and so has the opponent's. You cannot change it.\n- (ignore) place_fleet(ships):")
         .replace("fire(target, request_id) shoots ONE cell", "fire shoots ONE cell"))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        cmd = self.path.strip("/")
        if cmd == "rules":
            ans = {"rules": RULES}
        elif cmd == "observe":
            ans = svc.observe()
        elif cmd == "place":
            ans = {"status": "rejected", "message": "fleets are dealt by the server in this game; just fire",
                   "observation": svc.observe()}
        elif cmd == "fire":
            ans = svc.fire(body.get("target"), body.get("request_id"))
        elif cmd == "admin" and body.get("key") == key:
            ans = {**svc.game.result(), "actions": svc.actions, "player_fleet": fleet_names(mine),
                   "opponent_fleet": fleet_names(enemy)}
        else:
            ans = {"status": "unknown_command"}
        data = json.dumps(ans).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


print("ready", port, "game", game, flush=True)
HTTPServer(("127.0.0.1", port), H).serve_forever()
