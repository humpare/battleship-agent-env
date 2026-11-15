"""Exact reference versus runbooks and belief-based approximations (no model calls)."""
import sys, time
from incident.kernel import Params
from incident.solver import Exact
from incident import controllers as C
sys.setrecursionlimit(10000)
p = Params(horizon=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
rows = [("idle (always wait)", C.idle)]
for w in (0, 1, 2):
    rows += [(f"runbook wait {w}, verify, check lag", C.make_runbook(True, w)),
             (f"runbook wait {w}, no verify, check lag", C.make_runbook(False, w)),
             (f"runbook wait {w}, no verify, no lag check", C.make_runbook(False, w, check_lag=False, restart_if_lag=False)),
             (f"runbook wait {w}, verify, lag check, no restart", C.make_runbook(True, w, restart_if_lag=False))]
rows += [("QMDP", C.make_qmdp(p)), ("lookahead 1 + QMDP", C.make_lookahead(p, 1)), ("lookahead 2 + QMDP", C.make_lookahead(p, 2))]
res = []
for name, c in rows:
    t = time.time(); v = C.evaluate(p, c); res.append((name, v)); print(f"{name:48s} {v:7.3f}  ({time.time()-t:.0f}s)", flush=True)
t = time.time(); ex = Exact(p); vstar = ex.value(); print(f"{'EXACT':48s} {vstar:7.3f}  ({time.time()-t:.0f}s)", flush=True)
v0 = res[0][1]
print("\nnormalised (idle = 0, exact = 1)")
for name, v in sorted(res, key=lambda x: -x[1]):
    print(f"  {name:48s} {(v - v0) / (vstar - v0):6.3f}")
