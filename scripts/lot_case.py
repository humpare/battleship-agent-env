"""Write one static case (all 12 assay results shown) as text, plus the exact answer."""
import json, random, sys
from lotrelease.pilot import gen_random, Solver

seed = int(sys.argv[1])
inst = gen_random(seed, pool=(8, 24), width=(3, 6), max_assays=12, budget=18)
sv = Solver(inst)
rng = random.Random(1000 + seed)
w = rng.randrange(len(sv.worlds))
sig = sv.world_sig[w]
s = sv.root
for a in range(sv.m):
    s = sv.child(s, a, (sig >> (sv.m - 1 - a)) & 1)
S = lambda i: f"S{i + 1:02d}"
cert = [i for i in range(len(inst.products)) if not (int(sv.unsafe[s]) >> i) & 1]
cleared = sv.naive_cleared(s)
naive = [i for i, pb in enumerate(sv.prod_bits) if pb & ~cleared == 0]
lines = ["CONTAMINATION INCIDENT: LOT RELEASE DECISION", "",
         "There are 64 source lots, S01 to S64. EXACTLY TWO of them are contaminated; every pair is equally likely beforehand.",
         "A pooled assay mixes samples from the listed source lots. It is POSITIVE if at least one lot in the pool is contaminated, otherwise NEGATIVE. Assays are fully reliable.",
         "A product lot is unsafe if any of the source lots it was made from is contaminated.",
         "You may release a product lot only if the assay results PROVE it is safe, that is, it is safe under every pair of contaminated source lots that is consistent with all results.",
         "If you release even one product lot that is not proven safe, the whole submission scores zero. Otherwise your score is the value you release divided by the value that is truly safe.", "",
         "ASSAY RESULTS"]
for k, p in enumerate(inst.pools):
    res = "POSITIVE" if (sig >> (sv.m - 1 - k)) & 1 else "NEGATIVE"
    lines.append(f"A{k + 1:02d} {res}: " + " ".join(S(i) for i in sorted(p)))
lines += ["", "PRODUCT LOTS (name, value, source lots used)"]
for i, (p, v) in enumerate(zip(inst.products, inst.value)):
    lines.append(f"P{i + 1:02d} value {v}: " + " ".join(S(x) for x in p))
lines += ["", "Work this out by reasoning only. Finish with one line in exactly this form:", "RELEASE: P01 P02 ..."]
open(f"results/lot-pilot/case{seed}.txt", "w").write("\n".join(lines) + "\n")
val = lambda ids: sum(inst.value[i] for i in ids)
json.dump({"seed": seed, "truth": [S(x) for x in sv.worlds[w]], "certified": [f"P{i + 1:02d}" for i in cert],
           "naive": [f"P{i + 1:02d}" for i in naive], "values": {f"P{i + 1:02d}": v for i, v in enumerate(inst.value)},
           "safe_value": sv.world_safe[w], "certified_value": val(cert), "naive_value": val(naive),
           "worlds_left": round(float(sv.mass[s]) * len(sv.worlds))},
          open(f"results/lot-pilot/case{seed}.json", "w"), indent=1)
print(seed, "worlds left", round(float(sv.mass[s]) * 2016), "certified", len(cert), val(cert), "naive", len(naive), val(naive), "safe", sv.world_safe[w])
