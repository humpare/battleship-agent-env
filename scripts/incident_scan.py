"""Scan model parameters: where does the exact reference clearly beat QMDP and the runbooks?"""
import itertools, sys
from multiprocessing import Pool
from incident.kernel import Params
from incident.solver import Exact
from incident import controllers as C

def run(cfg):
    sys.setrecursionlimit(10000)
    p = Params(horizon=7, **cfg)
    v0 = C.evaluate(p, C.idle)
    books = [C.evaluate(p, C.make_runbook(v, w, cl, rs)) for v in (True, False) for w in (0, 1, 2)
             for cl, rs in ((True, True), (True, False), (False, False))]
    restart_first = C.evaluate(p, lambda t, b, h: C.A["wait"] if C._healthy(h) else C.A["restart_a"])
    qm = C.evaluate(p, C.make_qmdp(p)); la = C.evaluate(p, C.make_lookahead(p, 1))
    vs = Exact(p).value(); n = lambda v: (v - v0) / (vs - v0)
    return cfg, v0, vs, n(max(books)), n(restart_first), n(qm), n(la)

if __name__ == "__main__":
    grid = []
    for fixes, rt, fates, lagp, sr, sb in itertools.product(
            [(1.0, 0.9, 0.7, 0.5), (1.0, 0.5, 0.3, 0.1), (1.0, 0.3, 0.0, 0.0)], [2, 3],
            [(0.60, 0.15, 0.15, 0.10), (0.35, 0.25, 0.25, 0.15)], [0.3, 0.6], [0.2, 0.5], [8.0, 20.0]):
        grid.append(dict(restart_fixes=fixes, restart_ticks=rt, fates=fates, prior_lag=lagp, slow_recover=sr, split_brain_cost=sb))
    print("idle  exact | best-runbook restart-loop QMDP lookahead1 | config", flush=True)
    with Pool(20) as pool:
        for cfg, v0, vs, b, rf, qm, la in pool.imap_unordered(run, grid):
            print(f"{v0:5.2f} {vs:5.2f} | {b:5.2f} {rf:5.2f} {qm:5.2f} {la:5.2f} | {list(cfg.values())}", flush=True)
