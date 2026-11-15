"""Coded player over MCP (the reference run). Needs only `mcp` and reference.py."""
import argparse
import asyncio
import json
import os
import uuid

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

try:
    from .reference import CheckerboardPlayer, DensityPlayer
except ImportError:
    from reference import CheckerboardPlayer, DensityPlayer


async def play(url: str, policy: str, max_calls: int | None):
    player = DensityPlayer() if policy == "reference" else CheckerboardPlayer()
    async with streamable_http_client(url) as s:
        async with ClientSession(s[0], s[1]) as c:
            await c.initialize()
            call = lambda n, a=None: c.call_tool(n, a or {})  # noqa: E731
            json.loads((await call("get_rules")).content[0].text)
            obs = json.loads((await call("observe")).content[0].text)
            n = 0
            while not obs["game_over"] and policy != "nothing" and (max_calls is None or n < max_calls):
                r = json.loads((await call("fire", {"targets": player.choose(obs), "request_id": uuid.uuid4().hex})).content[0].text)
                obs, n = r["observation"], n + 1
    print(json.dumps({"calls": n, "shots_used": obs["shots_used"], "afloat": obs["ships_afloat_by_length"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://sim:8000/mcp")
    a = ap.parse_args()
    steps = os.environ.get("BATTLESHIP_MAX_CALLS")
    asyncio.run(play(a.url, os.environ.get("BATTLESHIP_POLICY", "reference"), int(steps) if steps else None))
