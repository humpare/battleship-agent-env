"""Duel scenario -> Harbor task."""
import argparse
import json
import shutil
from pathlib import Path

from dispatch.gen_tasks import COMPOSE, OPENCODE_CONFIG

from .duel import make_scenario
from .gen_task import MCP, TEST_SH

TESTS_DOCKERFILE = """\
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==1.26.4
COPY . /tests/
ENV PYTHONPATH=/tests BATTLESHIP_N=__N__
"""

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent

MAIN_DOCKERFILE = """\
# Agent container. Deliberately contains no game-server code and no scenario data.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bash \\
    && rm -rf /var/lib/apt/lists/*
# Used by the coded reference in solution/ (an MCP client); also there for an agent allowed to code.
RUN pip install --no-cache-dir mcp=={mcp_version} numpy==1.26.4
__TOOLS__
WORKDIR /app
"""
MCP_ONLY = """# The evaluated agent is a player: the game's MCP tools only, no shell, no file or web access.
ENV OPENCODE_CONFIG_CONTENT='{opencode_config}'"""
CODE_ALLOWED = "# Ablation condition: the agent keeps its normal tools (shell, files) and may write code."

INSTRUCTION = """\
# Battleship

You are playing Battleship on a {n}x{n} board against a coded opponent. A game server is
connected to you as the MCP server `battleship`. __ONLY__

Your job: sink the opponent's whole fleet before it sinks yours.

1. Call `get_rules` once and read it carefully: the rules differ from the usual game (ships may
   not touch), and they tell you exactly how the opponent places its fleet and chooses its shots.
2. Call `place_fleet` to place your own five ships.
3. Call `fire` with one target cell at a time. A hit lets you shoot again; a miss gives the
   opponent its turn, and you see its shots in the answer. Keep going until the observation
   says `game_over: true`.

There is one game and no reset. __TOOLS__
__HINT__"""
ONLY = {"mcp": "It is the only thing you can act on.",
        "code": "The game can only be played through it."}
TOOLS = {"mcp": "You have no other tools and need none.",
         "code": "You may also use your shell: Python 3.12 with numpy and the `mcp` client package is "
                 "installed, and the game server's MCP endpoint is http://sim:8000/mcp (streamable HTTP). "
                 "Writing code to work out your moves is allowed."}
HINT = """
## Advice from a strong player

- Because ships may not touch, most legal fleets sit in the corners and along the edges. On this
  board a corner holds a ship in about 61% of legal fleets, the centre in about 52%, and the four
  cells diagonally next to a corner (B2, D2, B4, D4) in about 3%. Do not hunt on a diagonal or
  checkerboard pattern out of habit.
- Before every shot, work out which legal fleets still agree with everything you have seen, and
  shoot the cell that holds a ship in the most of them. Remember that all cells around a sunk
  ship are water.
- The opponent searches the same way against you. Place your fleet where such a searcher looks
  late, and do not use an obvious layout such as ships lined up from A1.
"""

TASK_TOML = """\
schema_version = "1.3"

artifacts = [
    {{ source = "/data/episode.jsonl", service = "sim" }},
]

[task]
name = "battleship/{id}"
description = "Play Battleship on a small board against a coded opponent that eliminates fleets exactly."
keywords = ["game", "hidden-information", "search", "mcp"]

[metadata]
category = "game"

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
timeout_sec = 1500.0

[environment]
build_timeout_sec = 600.0
cpus = 2
memory_mb = 3072
storage_mb = 4096

[[environment.mcp_servers]]
name = "battleship"
transport = "streamable-http"
url = "http://sim:8000/mcp"
"""

SIM_DOCKERFILE = f"""\
FROM python:3.12-slim
RUN pip install --no-cache-dir mcp=={MCP} numpy==1.26.4
COPY battleship /opt/app/battleship
COPY scenario.json /opt/app/scenario.json
ENV PYTHONPATH=/opt/app PYTHONUNBUFFERED=1 BATTLESHIP_N=__N__
RUN mkdir -p /data
CMD ["python", "-m", "battleship.duel_server", "--scenario", "/opt/app/scenario.json", "--log", "/data/episode.jsonl"]
"""

VERIFY = '''\
"""Recompute the result by replaying the game server's private action log."""
import json
from pathlib import Path

from battleship.duel import load_actions, replay, scenario_hash

out = Path("/logs/verifier")
sc = json.loads(Path("/tests/scenario.json").read_text())
log = Path("/data/episode.jsonl")
res = {"score": 0.0, "error": "no action log collected"}
if log.exists():
    try:
        header, actions = load_actions(log)
        if header and header.get("scenario_hash") == scenario_hash(sc):
            res = replay(sc, actions)
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
# Reference: exact elimination over every legal fleet, played through the same MCP tools.
set -euo pipefail
cd "$(dirname "$0")"
pip install --quiet --no-cache-dir numpy==1.26.4 >/dev/null 2>&1
export BATTLESHIP_N=__N__
python duel_client.py
"""


def write_task(sc: dict, out: Path, tools: str = "mcp", hint: bool = False) -> Path:
    d = out / sc["id"]
    if d.exists():
        shutil.rmtree(d)
    for sub in ("environment/sim/battleship", "tests/battleship", "solution"):
        (d / sub).mkdir(parents=True)
    n = str(sc.get("board", 5))
    (d / "instruction.md").write_text(INSTRUCTION.replace("{n}", n).replace("__ONLY__", ONLY[tools])
                                      .replace("__TOOLS__", TOOLS[tools]).replace("__HINT__", HINT if hint else ""))
    (d / "task.toml").write_text(TASK_TOML.format(id=sc["id"]))
    restrict = MCP_ONLY.format(opencode_config=json.dumps(OPENCODE_CONFIG, separators=(",", ":")))
    (d / "environment/Dockerfile").write_text(MAIN_DOCKERFILE.format(mcp_version=MCP).replace(
        "__TOOLS__", restrict if tools == "mcp" else CODE_ALLOWED))
    (d / "environment/docker-compose.yaml").write_text(COMPOSE)
    (d / "environment/sim/Dockerfile").write_text(SIM_DOCKERFILE.replace("__N__", n))
    (d / "tests/Dockerfile").write_text(TESTS_DOCKERFILE.replace("__N__", n))
    for target in (d / "environment/sim", d / "tests"):
        (target / "scenario.json").write_text(json.dumps(sc))
        (target / "battleship" / "__init__.py").write_text("")
        shutil.copy(PKG / "duel.py", target / "battleship" / "duel.py")
        shutil.copy(PKG / "duel_reference.py", target / "battleship" / "duel_reference.py")
    shutil.copy(PKG / "duel_server.py", d / "environment/sim/battleship/duel_server.py")
    (d / "tests/test.sh").write_text(TEST_SH)
    (d / "tests/verify.py").write_text(VERIFY)
    (d / "solution/solve.sh").write_text(SOLVE.replace("__N__", n))
    shutil.copy(PKG / "duel_reference.py", d / "solution/duel_reference.py")
    shutil.copy(PKG / "duel_client.py", d / "solution/duel_client.py")
    for table in (PKG / f"placement_{n}.json", PKG / f"placement_{n}_duel.json"):
        if table.exists():
            shutil.copy(table, d / "solution" / table.name)
    # the yardstick classes import from the package; the reference client only needs ExactPlayer
    (d / "tests/test.sh").chmod(0o755)
    (d / "solution/solve.sh").chmod(0o755)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--opponent", default="exact", choices=["exact", "checkerboard"])
    ap.add_argument("--out", default="battleship")
    ap.add_argument("--tools", default="mcp", choices=["mcp", "code"],
                    help="mcp: the agent has only the game tools (the judged setting); code: it may also run code")
    ap.add_argument("--hint", action="store_true", help="add a strategy primer to the instruction")
    a = ap.parse_args()
    from .duel import N
    for seed in a.seeds:
        sc = make_scenario(f"bs{N}-s{seed}", seed, a.opponent, uniform=True, disclose_opponent=True)
        (ROOT / "scenarios" / "battleship").mkdir(parents=True, exist_ok=True)
        (ROOT / "scenarios" / "battleship" / f"{sc['id']}.json").write_text(json.dumps(sc))
        print("wrote", write_task(sc, ROOT / "tasks" / a.out, a.tools, a.hint))
