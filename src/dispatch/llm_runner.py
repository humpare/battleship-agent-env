"""Run a language model as dispatcher against one in-process episode, without Docker.

This is the fast path for calibration and for evaluating small models. It exposes the
same three tools, the same rules and the same scoring (replay of the action log) as the
Harbor task; only the transport differs. Official results come from Harbor.

Modes
  agent      multi-turn tool calling with full history (what OpenCode does)
  stateless  one prompt per decision: rules + network + current observation -> JSON commands.
             The task is Markov in the observation, so this loses nothing in principle.

Usage
  python -m dispatch.llm_runner --scenario scenarios/dev/a-basic-s1.json --mode stateless \
      --base-url http://localhost:11434/v1 --model some-model --out runs/llm/name
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import httpx

from .gen_tasks import INSTRUCTION
from .service import EpisodeService, load_batches
from .sim import replay

TOOLS = [
    {"type": "function", "function": {
        "name": "get_rules", "description": "Task objective, rules, command format, timing, scoring, the network and the timetable.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "observe", "description": "Current state of every train. Does not advance time.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "step",
        "description": "Apply a batch of train commands (may be empty), then advance time until the next event. Returns the new observation.",
        "parameters": {"type": "object", "required": ["commands", "expected_decision_id", "request_id"], "properties": {
            "commands": {"type": "array", "items": {"type": "object"},
                         "description": 'e.g. [{"train":"T1","action":"route","route":["W_dep","S1","L1a"]},{"train":"T2","action":"hold"}]'},
            "expected_decision_id": {"type": "integer"},
            "request_id": {"type": "string"}}}}},
]

STATELESS_SYSTEM = """\
{rules}

NETWORK
{network}

You decide one step at a time. You are given the current observation. Reply with ONLY a JSON
list of commands for this step (an empty list [] if no train needs a new order). No other text.
"""


def slim_observation(obs: dict) -> dict:
    """The observation with static and empty fields dropped, to keep prompts short."""
    trains = []
    for t in obs["trains"]:
        if t["status"] == "arrived":
            continue
        s = {"id": t["id"], "status": t["status"], "from": t["origin"], "to": t["destination"],
             "dep": t["earliest_departure"], "deadline": t["deadline"]}
        if t["location"]:
            s["at"] = t["location"]
        if t["route"]:
            s["route"] = t["route"]
        for k in ("holding", "waiting_at_end_of_route", "blocked_by"):
            if t[k]:
                s[k] = t[k]
        if t["malfunction_ticks_remaining"]:
            s["malfunction_ticks"] = t["malfunction_ticks_remaining"]
        trains.append(s)
    return {"time": obs["time"], "horizon": obs["horizon"], "trains": trains,
            "events": [{k: v for k, v in e.items() if k != "tick"} for e in obs["events_since_last_decision"]]}


def stateless_messages(rules: dict, obs: dict) -> list[dict]:
    system = STATELESS_SYSTEM.format(rules=rules["rules"], network=json.dumps(rules["network"], separators=(",", ":")))
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(slim_observation(obs), separators=(",", ":"))}]


def parse_commands(text: str):
    """First JSON list in the reply, or None."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    start = text.find("[")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "["
            depth -= text[i] == "]"
            if depth == 0:
                try:
                    val = json.loads(text[start:i + 1])
                    if isinstance(val, list):
                        return val
                except ValueError:
                    pass
                break
        start = text.find("[", start + 1)
    return None


class Chat:
    def __init__(self, base_url: str, model: str, api_key: str = "none", temperature: float = 0.0,
                 max_tokens: int = 4096, timeout: float = 600.0, extra: dict | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.extra = extra or {}
        self.http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})
        self.calls = self.prompt_tokens = self.completion_tokens = self.retries = 0
        self.seconds = 0.0

    def __call__(self, messages: list[dict], tools: list | None = None) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens, **self.extra}
        if tools:
            body["tools"] = tools
        t0 = time.perf_counter()
        for attempt in range(12):  # rate limits and transient server errors are not the model's fault
            try:
                r = self.http.post(self.url, json=body)
            except httpx.TransportError:
                r = None
            if r is not None and r.status_code not in (408, 409, 429, 500, 502, 503, 504, 529):
                break
            self.retries += 1
            time.sleep(min(90.0, 3.0 * 1.7 ** attempt))
        self.seconds += time.perf_counter() - t0
        if r is None:
            raise RuntimeError("model API unreachable after retries")
        r.raise_for_status()
        data = r.json()
        self.calls += 1
        usage = data.get("usage") or {}
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        return data["choices"][0]["message"]

    def metrics(self) -> dict:
        return {"model": self.model, "llm_calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "llm_seconds": round(self.seconds, 1),
                "api_retries": self.retries}


def run_stateless(svc: EpisodeService, chat, transcript: list, max_steps: int | None = None) -> dict:
    rules, obs = svc.get_rules(), svc.observe()
    refused = unparsable = steps = 0
    while not obs["episode_over"] and (max_steps is None or steps < max_steps):
        messages = stateless_messages(rules, obs)
        reply = chat(messages)
        text = reply.get("content") or ""
        commands = parse_commands(text)
        if commands is None:
            unparsable += 1
            commands = []
        ans = svc.step(commands, obs["decision_id"], f"s{steps}-a")
        if ans["status"] != "accepted":  # no second chance: an illegal order means no order this step
            refused += 1
            transcript.append({"decision": obs["decision_id"], "reply": text, "refused": ans["message"]})
            ans = svc.step([], obs["decision_id"], f"s{steps}-b")
        else:
            transcript.append({"decision": obs["decision_id"], "reply": text})
        obs = ans["observation"]
        steps += 1
    return {"mode": "stateless", "steps": steps, "refused_batches": refused, "unparsable_replies": unparsable}


def run_agent(svc: EpisodeService, chat, transcript: list, max_turns: int = 400) -> dict:
    messages = [{"role": "system", "content": "You are a careful railway dispatcher. Use the tools."},
                {"role": "user", "content": INSTRUCTION}]
    tool_calls = refused = turns = idle = 0
    while turns < max_turns:
        turns += 1
        reply = chat(messages, TOOLS)
        messages.append({k: v for k, v in reply.items() if k in ("role", "content", "tool_calls")})
        transcript.append({"turn": turns, "content": reply.get("content"), "tool_calls": reply.get("tool_calls")})
        calls = reply.get("tool_calls") or []
        if not calls:
            idle += 1
            if svc.sim.over or idle >= 2:
                break  # the model has stopped acting; standing orders run out in the verifier
            messages.append({"role": "user", "content": "Continue dispatching with the tools until episode_over is true."})
            continue
        idle = 0
        for c in calls:
            tool_calls += 1
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except ValueError:
                args = None
            if name == "get_rules":
                out = svc.get_rules()
            elif name == "observe":
                out = svc.observe()
            elif name == "step" and isinstance(args, dict):
                out = svc.step(args.get("commands"), args.get("expected_decision_id"), args.get("request_id"))
                refused += out["status"] not in ("accepted",)
            else:
                out = {"status": "invalid_request", "message": f"unknown tool or unreadable arguments: {name}"}
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": json.dumps(out, separators=(",", ":"))})
        if svc.sim.over:
            break
    return {"mode": "agent", "turns": turns, "tool_calls": tool_calls, "refused_batches": refused}


def run(scenario: dict, chat, mode: str, out_dir: Path, max_steps: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "episode.jsonl"
    log.unlink(missing_ok=True)
    svc = EpisodeService(scenario, log)
    transcript: list = []
    t0 = time.perf_counter()
    error = None
    try:
        info = run_stateless(svc, chat, transcript, max_steps) if mode == "stateless" else run_agent(svc, chat, transcript)
    except Exception as e:  # a model/API failure ends the model's play; obligations remain
        info, error = {"mode": mode}, f"{type(e).__name__}: {e}"
    _, batches = load_batches(log)
    result = replay(scenario, batches)  # the same scoring path as the Harbor verifier
    result.update(info)
    result["wall_seconds"] = round(time.perf_counter() - t0, 1)
    result["error"] = error
    if hasattr(chat, "metrics"):
        result.update(chat.metrics())
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    (out_dir / "transcript.jsonl").write_text("\n".join(json.dumps(t) for t in transcript))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, nargs="+")
    ap.add_argument("--mode", default="stateless", choices=["stateless", "agent"])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="LLM_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--extra", default="{}", help="extra JSON fields for the chat request")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for path in a.scenario:
        scenario = json.loads(Path(path).read_text())
        chat = Chat(a.base_url, a.model, os.environ.get(a.api_key_env, "none"), a.temperature, a.max_tokens,
                    extra=json.loads(a.extra))
        res = run(scenario, chat, a.mode, Path(a.out) / scenario["id"], a.max_steps)
        print(json.dumps({k: res.get(k) for k in ("scenario", "score", "completed", "trains", "total_lateness",
                                                   "refused_batches", "unparsable_replies", "steps", "turns",
                                                   "llm_calls", "llm_seconds", "error")}))


if __name__ == "__main__":
    main()
