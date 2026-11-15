"""Play a small model (optionally with a LoRA adapter) against the exact opponent on boards
it was never trained on, through the real game service. All games run side by side, so each
round is one batched generation.

Usage: train/.venv/bin/python train/bs_eval.py [--adapter train/out/bs5-v1] --out results/small-model/base.json
Reports per board group: mean score, wins, share of refused answers, and shot quality
(the chosen cell's chance of a hit divided by the best available chance, from the exact count).
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from bs_common import PLACE, eval_scenarios, messages, read_fleet, read_shot, shot_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

from battleship.duel import DuelService
from battleship.duel_reference import ExactPlayer, N, read

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--adapter", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=65)
ap.add_argument("--groups", nargs="+", default=["frozen30", "extra100"])
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.base)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": 0})
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.eval()


def generate(prompts: list[str], max_new: int) -> list[str]:
    out = []
    for i in range(0, len(prompts), a.batch):
        texts = [tok.apply_chat_template(messages(p), tokenize=False, add_generation_prompt=True) for p in prompts[i:i + a.batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


judge = ExactPlayer()


def quality(obs: dict, target: str) -> float:
    """Chance of a hit at the chosen cell divided by the best available chance (exact count)."""
    left = judge.consistent(obs["opponent_board"])
    seen = read(obs["opponent_board"])
    shot = set(seen["o"]) | set(seen["x"]) | set(seen["#"])
    share = {(r, c): float(((left >> np.uint64(r * N + c)) & np.uint64(1)).mean())
             for r in range(N) for c in range(N) if (r, c) not in shot}
    cell = (int(target[1:]) - 1, "ABCDEFGHI".index(target[0]))
    return share.get(cell, 0.0) / max(share.values())


report = {"base": a.base, "adapter": a.adapter, "groups": {}}
for group, scs in eval_scenarios().items():
    if group not in a.groups:
        continue
    games = [{"svc": DuelService(sc), "refused": 0, "decisions": 0, "q": [], "alive": True, "first": None} for sc in scs]
    # placement: up to 3 tries, then the game is abandoned (scores 0, like doing nothing)
    for _ in range(3):
        todo = [g for g in games if g["svc"].game.mine is None]
        for g, text in zip(todo, generate([PLACE] * len(todo), 80)):
            g["decisions"] += 1
            if g["svc"].place_fleet(read_fleet(text))["status"] != "accepted":
                g["refused"] += 1
    import random
    from battleship.duel import fleet_names, uniform_fleet
    for i, g in enumerate(games):
        g["placed"] = g["svc"].game.mine is not None
        if not g["placed"]:  # could not place a legal fleet: deal one at random so shooting can still be measured
            g["svc"].place_fleet(fleet_names(uniform_fleet(random.Random(i))))
        g["fleet"] = g["svc"].actions[0]["ships"] if g["placed"] else None
    while True:
        todo = [g for g in games if g["alive"] and not g["svc"].game.over]
        if not todo:
            break
        obs = [g["svc"].observe() for g in todo]
        for g, o, text in zip(todo, obs, generate([shot_prompt(o) for o in obs], 6)):
            g["decisions"] += 1
            target = read_shot(text)
            r = g["svc"].fire(target, f"s{g['decisions']}") if target else {"status": "rejected"}
            if r["status"] != "accepted":
                g["refused"] += 1
                if g["refused"] >= 5:  # gives up: the game stays unfinished
                    g["alive"] = False
                continue
            g["first"] = g["first"] or target
            g["q"].append(quality(o, target))
    res = [g["svc"].game.result() for g in games]
    report["groups"][group] = {
        "boards": len(games),
        # strict: a game in which the model never produced a legal fleet scores 0 (as in the task)
        "mean_score": round(statistics.mean(r["score"] if g["placed"] else 0.0 for g, r in zip(games, res)), 4),
        "mean_score_fleet_dealt_if_needed": round(statistics.mean(r["score"] for r in res), 4),
        "placed_own_fleet": sum(g["placed"] for g in games),
        "wins": sum(r["won"] and g["placed"] for g, r in zip(games, res)), "finished": sum(r["finished"] for r in res),
        "refused_share": round(sum(g["refused"] for g in games) / max(1, sum(g["decisions"] for g in games)), 4),
        "shot_quality": round(statistics.mean(q for g in games for q in g["q"]), 4) if any(g["q"] for g in games) else None,
        "first_shots": {c: sum(g["first"] == c for g in games) for c in sorted({g["first"] for g in games if g["first"]})},
        "distinct_fleets": len({json.dumps(g["fleet"]) for g in games if g["fleet"]}),
        "scores": [r["score"] for r in res]}
    print(group, {k: v for k, v in report["groups"][group].items() if k != "scores"}, flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(report, indent=1))
