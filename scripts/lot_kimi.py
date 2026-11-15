"""Ask the Verda-hosted model for the release list of each static case; grade against the exact answer."""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
env = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l and not l.startswith("#"))
MODEL = sys.argv[1]

def grade(seed, text):
    a = json.load(open(f"results/lot-pilot/case{seed}.json"))
    m = re.findall(r"RELEASE:(.*)", text)
    rel = set(re.findall(r"P\d\d", m[-1])) if m else set()
    bad = sorted(rel - set(a["certified"]))
    v = sum(a["values"][p] for p in rel)
    return {"seed": seed, "released": len(rel), "uncertified": bad, "score": 0.0 if bad or not m else v / a["safe_value"],
            "reference": a["certified_value"] / a["safe_value"], "naive": a["naive_value"] / a["safe_value"]}

def run(seed):
    body = {"model": MODEL, "messages": [{"role": "user", "content": open(f"results/lot-pilot/case{seed}.txt").read()}], "max_tokens": 32000}
    req = urllib.request.Request(env["VERDA_BASE_URL"] + "/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + env["VERDA_API_KEY"], "Content-Type": "application/json"})
    t = time.time()
    for attempt in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=1500)); break
        except Exception as e:
            err = str(e)[:200]; time.sleep(20)
    else:
        return {"seed": seed, "error": err}
    text = r["choices"][0]["message"].get("content") or ""
    open(f"results/lot-pilot/{MODEL}-case{seed}.txt", "w").write(text)
    g = grade(seed, text); g["seconds"] = round(time.time() - t); g["tokens"] = r.get("usage", {}).get("completion_tokens")
    return g

if __name__ == "__main__":
    with ThreadPoolExecutor(3) as ex:
        for g in ex.map(run, [int(x) for x in sys.argv[2:]]):
            print(json.dumps(g), flush=True)
