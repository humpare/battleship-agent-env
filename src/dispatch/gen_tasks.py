"""Turn scenario files into self-contained Harbor task directories.

Usage: python -m dispatch.gen_tasks [scenario.json ...] [--out tasks]   (default: every scenarios/<split>/*.json)

Layout of one task:
  environment/Dockerfile          agent container: no simulator code, MCP-only OpenCode config
  environment/docker-compose.yaml adds the `sim` sidecar
  environment/sim/                sidecar image: simulator + MCP server + this scenario
  tests/                          separate verifier image: replays the sidecar's action log
  solution/                       coded reference, plays through the same MCP tools
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
SIM_FILES = ["__init__.py", "layout.py", "sim.py", "rules.py", "service.py", "server.py"]

# The evaluated agent is a dispatcher, not a programmer: it gets the three MCP
# tools and a scratch to-do list, and nothing else.
OPENCODE_CONFIG = {
    "tools": {name: False for name in (
        "bash", "edit", "write", "patch", "multiedit", "read", "grep", "glob", "list",
        "webfetch", "websearch", "task", "skill", "lsp", "codesearch", "google_search")},
    "permission": {"bash": "deny", "edit": "deny", "webfetch": "deny", "websearch": "deny"},
}

INSTRUCTION = """\
# Railway dispatching

You are the dispatcher for a small railway. A simulator is connected to you as the MCP
server `dispatch`. It is the only thing you can act on.

Your job: bring every scheduled train to its destination, as close to on time as you can,
before the episode ends.

How to work:

1. Call `get_rules` once. It gives you the full rules, the command format, the track network
   and the timetable. Read it carefully: the simulator does not stop you from sending trains
   into each other.
2. Call `observe` to see the current state.
3. Call `step` with your commands. Time only moves when you call `step`. Each call returns the
   new state. Keep going until the observation says `episode_over: true`.

There is one episode and no reset. Whatever is unfinished when you stop, or when the episode
ends, counts against you. You have no other tools and need none.
"""

TASK_TOML = """\
schema_version = "1.3"

artifacts = [
    {{ source = "/data/episode.jsonl", service = "sim" }},
]

[task]
name = "flatland-dispatch/{id}"
description = "Dispatch trains on a single-track railway with passing loops ({stage})."
keywords = ["dispatching", "planning", "flatland", "mcp"]

[metadata]
category = "planning"
stage = "{stage}"
split = "{split}"
trains = {trains}
horizon_ticks = {horizon}

[verifier]
timeout_sec = 300.0
environment_mode = "separate"

[verifier.environment]
network_mode = "no-network"
build_timeout_sec = 1200.0
cpus = 1
memory_mb = 2048
storage_mb = 10240

[agent]
timeout_sec = {agent_timeout}.0

[environment]
build_timeout_sec = 1200.0
cpus = 1
memory_mb = 2048
storage_mb = 10240

[[environment.mcp_servers]]
name = "dispatch"
transport = "streamable-http"
url = "http://sim:8000/mcp"
"""

MAIN_DOCKERFILE = """\
# Agent container. Deliberately contains no simulator code and no scenario data.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bash \\
    && rm -rf /var/lib/apt/lists/*
# Only needed by the coded reference in solution/ (an MCP client).
RUN pip install --no-cache-dir mcp=={mcp_version}
# The evaluated agent is a dispatcher: MCP tools only, no shell, no file or web access.
ENV OPENCODE_CONFIG_CONTENT='{opencode_config}'
WORKDIR /app
"""

COMPOSE = """\
# Merged on top of Harbor's base compose file. `main` is the agent container.
services:
  main:
    depends_on:
      sim:
        condition: service_healthy

  sim:
    build:
      context: ./sim
    expose:
      - "8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import socket; socket.create_connection(('localhost', 8000), timeout=2).close()"]
      interval: 2s
      timeout: 5s
      retries: 30
      start_period: 5s
"""

SIM_DOCKERFILE = """\
# Simulator sidecar: owns the episode, the scenario (including hidden events) and the action log.
FROM python:3.12-slim
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY dispatch /opt/app/dispatch
COPY scenario.json /opt/app/scenario.json
ENV PYTHONPATH=/opt/app PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
RUN mkdir -p /data
CMD ["python", "-m", "dispatch.server", "--scenario", "/opt/app/scenario.json", "--log", "/data/episode.jsonl"]
"""

TESTS_DOCKERFILE = """\
# Verifier image (separate from the agent's container, no network).
FROM python:3.12-slim
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY . /tests/
ENV PYTHONPATH=/tests PYTHONWARNINGS=ignore
"""

TEST_SH = """\
#!/bin/bash
# Scores the episode by replaying the simulator sidecar's private action log.
mkdir -p /logs/verifier
python /tests/verify.py --scenario /tests/scenario.json --log /data/episode.jsonl --out /logs/verifier
"""

VERIFY_PY = '''\
"""Verifier: recompute the score from the action log. Trusts nothing the agent wrote."""
import argparse
import json
import sys
from pathlib import Path

from dispatch.service import load_batches
from dispatch.sim import replay, scenario_hash

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def finish(score: float, detail: dict) -> None:
    (out / "reward.txt").write_text(f"{score:.6f}")
    (out / "result.json").write_text(json.dumps(detail, indent=2))
    print(json.dumps({k: v for k, v in detail.items() if k != "per_train"}))
    sys.exit(0)


scenario = json.loads(Path(a.scenario).read_text())
log = Path(a.log)
if not log.exists():
    finish(0.0, {"score": 0.0, "error": "no action log collected from the simulator"})
try:
    header, batches = load_batches(log)
except Exception as e:  # unreadable log: no credit
    finish(0.0, {"score": 0.0, "error": f"action log unreadable: {e}"})
if header is None or header.get("scenario_hash") != scenario_hash(scenario):
    finish(0.0, {"score": 0.0, "error": "action log does not belong to this scenario"})
try:
    result = replay(scenario, batches)
except Exception as e:  # a log the simulator itself would never have accepted
    finish(0.0, {"score": 0.0, "error": f"action log does not replay: {e}"})
result["accepted_steps"] = len(batches)
finish(result["score"], result)
'''

SOLVE_SH = """\
#!/bin/bash
# Coded reference dispatcher. Plays through the same three MCP tools the agent gets.
set -euo pipefail
cd "$(dirname "$0")"
# DISPATCH_POLICY=nothing and DISPATCH_MAX_STEPS=N turn it into the do-nothing / stop-early baselines.
python mcp_client.py --url http://sim:8000/mcp --policy "${DISPATCH_POLICY:-reference}" \\
    ${DISPATCH_MAX_STEPS:+--max-steps "$DISPATCH_MAX_STEPS"}
"""


def requirements() -> str:
    out = subprocess.run(
        ["uv", "export", "--no-dev", "--no-hashes", "--no-emit-project", "--frozen"],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return "\n".join(line for line in out.splitlines() if line and not line.startswith("#")) + "\n"


def write_task(scenario: dict, out_dir: Path, reqs: str) -> Path:
    d = out_dir / scenario.get("meta", {}).get("split", "dev") / scenario["id"]
    if d.exists():
        shutil.rmtree(d)
    (d / "environment" / "sim" / "dispatch").mkdir(parents=True)
    (d / "tests" / "dispatch").mkdir(parents=True)
    (d / "solution").mkdir()
    meta = scenario.get("meta", {})
    mcp_version = next(line.split("==")[1] for line in reqs.splitlines() if line.startswith("mcp=="))

    (d / "instruction.md").write_text(INSTRUCTION)
    (d / "task.toml").write_text(TASK_TOML.format(
        id=scenario["id"], stage=meta.get("stage", "unspecified"), split=meta.get("split", "dev"),
        trains=len(scenario["trains"]), horizon=scenario["horizon"],
        agent_timeout=int(meta.get("agent_timeout_sec", 3600))))
    (d / "environment" / "Dockerfile").write_text(MAIN_DOCKERFILE.format(
        mcp_version=mcp_version, opencode_config=json.dumps(OPENCODE_CONFIG, separators=(",", ":"))))
    (d / "environment" / "docker-compose.yaml").write_text(COMPOSE)

    for target, dockerfile in ((d / "environment" / "sim", SIM_DOCKERFILE), (d / "tests", TESTS_DOCKERFILE)):
        (target / "Dockerfile").write_text(dockerfile)
        (target / "requirements.txt").write_text(reqs)
        (target / "scenario.json").write_text(json.dumps(scenario, indent=1))
        for f in SIM_FILES:
            shutil.copy(PKG / f, target / "dispatch" / f)
    (d / "tests" / "test.sh").write_text(TEST_SH)
    (d / "tests" / "verify.py").write_text(VERIFY_PY)
    (d / "solution" / "solve.sh").write_text(SOLVE_SH)
    for f in ("reference.py", "mcp_client.py"):
        shutil.copy(PKG / f, d / "solution" / f)
    for sh in (d / "tests" / "test.sh", d / "solution" / "solve.sh"):
        sh.chmod(0o755)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", help="scenario JSON files (default: scenarios/*.json)")
    ap.add_argument("--out", default=str(ROOT / "tasks"))
    a = ap.parse_args()
    files = [Path(p) for p in a.scenarios] or sorted((ROOT / "scenarios").glob("*/*.json"))
    reqs = requirements()
    for f in files:
        d = write_task(json.loads(f.read_text()), Path(a.out), reqs)
        print(f"wrote {d.relative_to(ROOT) if d.is_relative_to(ROOT) else d}")


if __name__ == "__main__":
    main()
