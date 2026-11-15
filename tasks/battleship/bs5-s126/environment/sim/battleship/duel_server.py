"""MCP server for one Battleship duel. Usage: python -m battleship.duel_server --scenario s.json --log /data/episode.jsonl"""
import argparse
import json

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from .duel import DuelService


def build(svc: DuelService) -> MCPServer:
    mcp = MCPServer("battleship", instructions="Battleship duel. get_rules, then place_fleet, then fire until game_over.")
    out = lambda d: json.dumps(d, separators=(",", ":"))  # noqa: E731

    @mcp.tool()
    def get_rules() -> str:
        """The full rules: board, fleets, setup, turns, scoring."""
        return out(svc.get_rules())

    @mcp.tool()
    def observe() -> str:
        """Both boards as you know them, ships afloat on each side, whose move it is. Changes nothing."""
        return out(svc.observe())

    @mcp.tool()
    def place_fleet(ships: list[list[str]] | str) -> str:
        """Place your own fleet once, before shooting. ships: five ships, each a list of cells,
        e.g. [["A1","A2","A3"],["C1","D1"],["F2","F3"],["A6"],["D5"]]."""
        return out(svc.place_fleet(ships))

    @mcp.tool()
    def fire(target: str, request_id: str) -> str:
        """Shoot one cell of the opponent's board, e.g. target="C5". Returns your result and, if you
        missed, every shot the opponent then fired at you. request_id: any string unique to this call."""
        return out(svc.fire(target, request_id))

    return mcp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    build(DuelService(json.load(open(a.scenario)), a.log)).run(
        "streamable-http", host="0.0.0.0", port=8000,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
