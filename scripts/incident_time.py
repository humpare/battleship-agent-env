import sys, time
from incident.kernel import Params
from incident.solver import Exact
sys.setrecursionlimit(10000)
for T in (6, 7, 8):
    t = time.time(); e = Exact(Params(horizon=T)); v = e.value()
    print(T, round(v, 4), len(e.memo), f"{time.time()-t:.1f}s", flush=True)
