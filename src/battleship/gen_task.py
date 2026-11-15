"""Scenario -> Harbor task (same shape as the railway tasks: sidecar game, replay verifier, MCP-only agent)."""
import argparse
import json
import shutil
from pathlib import Path

from dispatch.gen_tasks import COMPOSE, MAIN_DOCKERFILE, OPENCODE_CONFIG

from .game import make_scenario

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
MCP = "2.3.0"

INSTRUCTION = """\
# Battleship

A fleet is hidden on a 10x10 board. A game server is connected to you as the MCP server
`battleship`. It is the only thing you can act on.

Your job: sink the whole fleet using as few shots as you can.

1. Call `get_rules` once and read it carefully: the fleet composition and the rule that ships
   never touch tell you a lot about where ships can and cannot be.
2. Call `observe` to see the board.
3. Call `fire` with 1 to 5 target cells per call. Each call returns the result of every shot and
   the updated board. Keep going until the observation says `game_over: true`.

There is one game and no reset. You have no other tools and need none.
"""

TASK_TOML = """\
schema_version = "1.3"

artifacts = [
    {{ source = "/data/episode.jsonl", service = "sim" }},
]

[task]
name = "battleship/{id}"
description = "Sink a hidden fleet (classic Battleship rules) with a limited number of shots."
keywords = ["search", "hidden-information", "mcp"]

[metadata]
category = "search"
shot_budget = {budget}

[verifier]
timeout_sec = 120.0
environment_mode = "separate"

[verifier.environment]
network_mode = "no-network"
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 4096

[agent]
timeout_sec = 1800.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 4096

[[environment.mcp_servers]]
name = "battleship"
transport = "streamable-http"
url = "http://sim:8000/mcp"
"""

SIM_DOCKERFILE = f"""\
FROM python:3.12-slim
RUN pip install --no-cache-dir mcp=={MCP}
COPY battleship /opt/app/battleship
COPY scenario.json /opt/app/scenario.json
ENV PYTHONPATH=/opt/app PYTHONUNBUFFERED=1
RUN mkdir -p /data
CMD ["python", "-m", "battleship.server", "--scenario", "/opt/app/scenario.json", "--log", "/data/episode.jsonl"]
"""

TESTS_DOCKERFILE = """\
FROM python:3.12-slim
COPY . /tests/
ENV PYTHONPATH=/tests
"""

TEST_SH = """\
#!/bin/bash
mkdir -p /logs/verifier
python /tests/verify.py
"""

VERIFY = '''\
"""Recompute the result by replaying the game server's private shot log."""
import json
from pathlib import Path

from battleship.game import load_salvos, replay, scenario_hash

out = Path("/logs/verifier")
sc = json.loads(Path("/tests/scenario.json").read_text())
log = Path("/data/episode.jsonl")
res = {"score": 0.0, "error": "no shot log collected"}
if log.exists():
    try:
        header, salvos = load_salvos(log)
        if header and header.get("scenario_hash") == scenario_hash(sc):
            res = replay(sc, salvos)
        else:
            res = {"score": 0.0, "error": "log does not belong to this scenario"}
    except Exception as e:
        res = {"score": 0.0, "error": f"log does not replay: {e}"}
(out / "reward.txt").write_text(f"{res['score']:.6f}")
(out / "result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
'''

SOLVE = """\
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python client.py --url http://sim:8000/mcp
"""


def write_task(sc: dict, out: Path) -> Path:
    d = out / sc["id"]
    if d.exists():
        shutil.rmtree(d)
    for sub in ("environment/sim/battleship", "tests/battleship", "solution"):
        (d / sub).mkdir(parents=True)
    (d / "instruction.md").write_text(INSTRUCTION)
    (d / "task.toml").write_text(TASK_TOML.format(id=sc["id"], budget=sc["budget"]))
    (d / "environment/Dockerfile").write_text(MAIN_DOCKERFILE.format(
        mcp_version=MCP, opencode_config=json.dumps(OPENCODE_CONFIG, separators=(",", ":"))))
    (d / "environment/docker-compose.yaml").write_text(COMPOSE)
    (d / "environment/sim/Dockerfile").write_text(SIM_DOCKERFILE)
    (d / "tests/Dockerfile").write_text(TESTS_DOCKERFILE)
    for target in (d / "environment/sim", d / "tests"):
        (target / "scenario.json").write_text(json.dumps(sc))
        for f in ("__init__.py", "game.py", "server.py"):
            shutil.copy(PKG / f, target / "battleship" / f)
    (d / "tests/battleship/server.py").unlink()
    (d / "tests/test.sh").write_text(TEST_SH)
    (d / "tests/verify.py").write_text(VERIFY)
    (d / "solution/solve.sh").write_text(SOLVE)
    for f in ("reference.py", "client.py"):
        shutil.copy(PKG / f, d / "solution" / f)
    (d / "tests/test.sh").chmod(0o755)
    (d / "solution/solve.sh").chmod(0o755)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="battleship-poc")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--budget", type=int, default=100)
    a = ap.parse_args()
    sc = make_scenario(a.id, a.seed, a.budget)
    (ROOT / "scenarios" / "battleship").mkdir(parents=True, exist_ok=True)
    (ROOT / "scenarios" / "battleship" / f"{a.id}.json").write_text(json.dumps(sc))
    print("wrote", write_task(sc, ROOT / "tasks" / "battleship"))
