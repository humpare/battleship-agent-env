"""Run a coded policy against the MCP server, through the same three tools an agent gets.

This file and reference.py need only the ``mcp`` package: no simulator code.
Usage: python mcp_client.py --url http://sim:8000/mcp [--policy reference|nothing] [--max-steps N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

try:
    from .reference import ReferenceDispatcher
except ImportError:  # run as a plain script from solution/
    from reference import ReferenceDispatcher


async def call(session: ClientSession, name: str, args: dict | None = None) -> dict:
    res = await session.call_tool(name, args or {})
    return json.loads(res.content[0].text)


async def play(url: str, policy: str = "reference", max_steps: int | None = None) -> dict:
    t0 = time.perf_counter()
    async with streamable_http_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            tools = sorted(t.name for t in (await session.list_tools()).tools)
            rules = await call(session, "get_rules")
            obs = await call(session, "observe")
            ref = ReferenceDispatcher(rules["network"])
            steps = 0
            while not obs["episode_over"] and (max_steps is None or steps < max_steps):
                commands = ref.decide(obs) if policy == "reference" else []
                ans = await call(session, "step", {
                    "commands": commands, "expected_decision_id": obs["decision_id"],
                    "request_id": uuid.uuid4().hex})
                if ans["status"] != "accepted":
                    raise RuntimeError(f"step refused: {ans['status']}: {ans['message']}")
                obs = ans["observation"]
                steps += 1
    arrived = sum(t["status"] == "arrived" for t in obs["trains"])
    return {"tools": tools, "steps": steps, "final_tick": obs["time"], "arrived": arrived,
            "trains": len(obs["trains"]), "seconds": round(time.perf_counter() - t0, 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://sim:8000/mcp")
    ap.add_argument("--policy", default="reference", choices=["reference", "nothing"])
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(asyncio.run(play(a.url, a.policy, a.max_steps))))
