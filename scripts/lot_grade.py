import json, re, sys
a = json.load(open(f"results/lot-pilot/case{sys.argv[1]}.json"))
rel = set(re.findall(r"P\d\d", sys.argv[2]))
bad = sorted(rel - set(a["certified"])); v = sum(a["values"][p] for p in rel)
print("truth", a["truth"], "released", len(rel), "uncertified", bad, "score", 0 if bad else round(v / a["safe_value"], 3),
      "reference", round(a["certified_value"] / a["safe_value"], 3), "missed", sorted(set(a["certified"]) - rel))
