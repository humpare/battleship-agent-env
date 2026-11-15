"""Scan generator settings: mean over seeds of exact, best full-logic simple rule, best naive rule."""
import itertools, statistics as st
from lotrelease.pilot import gen_random, Solver, evaluate, POLICIES

rows = []
for pool, width, (ma, bud), nprod in itertools.product(
        [(6, 12), (8, 24), (16, 32), (20, 40)], [(1, 2), (2, 4), (3, 6)], [(6, 8), (9, 12), (12, 18)], [40]):
    ex, full, naive, inf = [], [], [], []
    for seed in range(1, 5):
        sv = Solver(gen_random(seed, pool=pool, width=width, max_assays=ma, budget=bud, n_prod=nprod))
        r = {n: evaluate(sv, p, rel) for n, p, rel in POLICIES}
        ex.append(r["exact DP"]["score"]); inf.append(r["exact DP"]["inferred_share"])
        full.append(max(v["score"] for k, v in r.items() if "full logic" in k))
        naive.append(max(v["score"] for k, v in r.items() if "naive release" in k))
    rows.append((st.mean(ex) - st.mean(naive), pool, width, ma, bud, st.mean(ex), st.mean(full), st.mean(naive), st.mean(inf)))
print("pool      width  assays/budget  exact  best-full  best-naive  inferred")
for g, pool, width, ma, bud, e, f, n, i in sorted(rows, reverse=True):
    print(f"{str(pool):9s} {str(width):7s} {ma:2d}/{bud:2d}        {e:.3f}  {f:.3f}      {n:.3f}       {i:.2f}")
