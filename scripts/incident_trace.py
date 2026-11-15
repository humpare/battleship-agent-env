"""Print what a controller does along its most likely observation branches."""
import sys
from incident.kernel import Params, ACTIONS, initial_belief
from incident.solver import expand
from incident import controllers as C
p = Params(); ctl = C.make_lookahead(p, 1)
def go(t, b, prob, path):
    if t >= p.horizon or prob < 0.03:
        print(f"{prob:5.2f}  " + "  ".join(path)); return
    a = ctl(t, b, [])
    r, succ = expand(p, b, a)
    for obs, (z, b2) in sorted(succ.items(), key=lambda x: -x[1][0]):
        o = obs[0] if isinstance(obs[0], str) else "/".join(map(str, obs[0]))
        go(t + 1, b2, prob * z, path + [f"{ACTIONS[a]}[{o}{'+' if obs[1] else ''}]"])
go(0, initial_belief(p), 1.0, [])
