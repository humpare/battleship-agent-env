"""The only command a model player may use.
  python3 play.py PORT rules
  python3 play.py PORT observe
  python3 play.py PORT place '[["A1","A2","A3"],["C1","D1"],["E2","E3"],["A5"],["C4"]]'
  python3 play.py PORT fire C4
  python3 play.py PORT next       (series only: start the next game)
  python3 play.py PORT history    (series only: every finished game in full)
"""
import json
import sys
import urllib.request
import uuid

port, cmd = sys.argv[1], sys.argv[2]
body = {}
if cmd == "place":
    body = {"ships": json.loads(sys.argv[3])}
elif cmd == "fire":
    body = {"target": sys.argv[3], "request_id": uuid.uuid4().hex}
req = urllib.request.Request(f"http://127.0.0.1:{port}/{cmd}", data=json.dumps(body).encode(), method="POST")
ans = json.loads(urllib.request.urlopen(req, timeout=120).read())
if cmd == "rules":
    print(ans["rules"])
elif cmd == "history":
    print(f"games finished: {ans['games_finished']} of {ans['games_total']}")
    for g in ans["history"]:
        print(f"--- game {g['game']}: winner {g['winner']}")
        print("  your fleet:     ", g["your_fleet"])
        print("  opponent fleet: ", g["opponent_fleet"])
        print("  your shots:     ", " ".join(g["your_shots"]))
        print("  opponent shots: ", " ".join(g["opponent_shots"]))
else:
    obs = ans.pop("observation", None) or (ans if cmd == "observe" else None)
    if cmd != "observe":
        print(json.dumps(ans))
    if obs:
        print("OPPONENT BOARD        YOUR BOARD")
        yours = obs.get("your_board") or [""] * len(obs["opponent_board"])
        for a, b in zip(obs["opponent_board"], yours):
            print(f"{a:22}{b}")
        print("opponent ships afloat:", obs["opponent_ships_afloat"], "| your ships afloat:", obs.get("your_ships_afloat"))
        print("next:", obs["next"], "| winner:", obs["winner"], "|", ans.get("series", ""))
