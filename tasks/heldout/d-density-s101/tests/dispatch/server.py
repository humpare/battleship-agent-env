"""MCP server (streamable HTTP) exposing exactly three tools for one episode.

Usage: python -m dispatch.server --scenario scenario.json --log /data/episode.jsonl
"""
from __future__ import annotations

import argparse
import json

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from .service import EpisodeService


def build(service: EpisodeService) -> MCPServer:
    mcp = MCPServer(
        "dispatch",
        instructions="Railway dispatching episode. Call get_rules first, then observe, then step repeatedly.",
    )

    def out(d: dict) -> str:
        return json.dumps(d, separators=(",", ":"))

    @mcp.tool()
    def get_rules() -> str:
        """Task objective, rules, command format, timing, scoring, the network and the timetable."""
        return out(service.get_rules())

    @mcp.tool()
    def observe() -> str:
        """Current state of every train. Does not advance time."""
        return out(service.observe())

    @mcp.tool()
    def step(commands: list[dict] | str, expected_decision_id: int, request_id: str) -> str:
        """Apply a batch of train commands (may be an empty list), then advance time until the
        next event (1 to a few ticks). Returns the new observation.
        commands: [{"train": "T1", "action": "route", "route": ["W_dep", "S1", "L1a"]},
                   {"train": "T2", "action": "hold"}]
        expected_decision_id: the decision_id from the latest observation.
        request_id: any string unique to this call."""
        return out(service.step(commands, expected_decision_id, request_id))

    return mcp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    with open(args.scenario) as f:
        scenario = json.load(f)
    service = EpisodeService(scenario, args.log)
    build(service).run(
        "streamable-http", host=args.host, port=args.port,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )


if __name__ == "__main__":
    main()
