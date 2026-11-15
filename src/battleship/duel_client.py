"""The reference run: the exact-elimination player over MCP, through the same four tools."""
import asyncio
import json
import os
import uuid

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from duel_reference import ExactPlayer


async def main(url="http://sim:8000/mcp"):
    max_shots = os.environ.get("BATTLESHIP_MAX_SHOTS")
    player = ExactPlayer(seed=int(os.environ.get("BATTLESHIP_PLACE_SEED", "1")))
    async with streamable_http_client(url) as s:
        async with ClientSession(s[0], s[1]) as c:
            await c.initialize()
            get = lambda n, a=None: c.call_tool(n, a or {})  # noqa: E731
            await get("get_rules")
            r = json.loads((await get("place_fleet", {"ships": player.place()})).content[0].text)
            obs, n = r["observation"], 0
            while not obs["game_over"] and (max_shots is None or n < int(max_shots)):
                r = json.loads((await get("fire", {"target": player.choose(obs), "request_id": uuid.uuid4().hex})).content[0].text)
                obs, n = r["observation"], n + 1
    print(json.dumps({"shots": n, "winner": obs["winner"]}))


asyncio.run(main())
