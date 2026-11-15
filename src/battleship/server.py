"""MCP server for one Battleship game. Usage: python -m battleship.server --scenario s.json --log /data/episode.jsonl"""
import argparse
import json

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from .game import Service


def build(svc: Service) -> MCPServer:
    mcp = MCPServer("battleship", instructions="Battleship. Call get_rules first, then observe, then fire until game_over.")
    out = lambda d: json.dumps(d, separators=(",", ":"))  # noqa: E731

    @mcp.tool()
    def get_rules() -> str:
        """The full rules, the fleet, the shot budget and how the game is scored."""
        return out(svc.get_rules())

    @mcp.tool()
    def observe() -> str:
        """The board as you know it, shots used and left, ships still afloat. Uses no shot."""
        return out(svc.observe())

    @mcp.tool()
    def fire(targets: list[str] | str, request_id: str) -> str:
        """Shoot 1 to 5 cells, in order, e.g. targets=["C7","E2"]. Returns miss/hit/sunk for each
        and the updated board. request_id: any string unique to this call."""
        return out(svc.fire(targets, request_id))

    return mcp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    svc = Service(json.load(open(a.scenario)), a.log)
    build(svc).run("streamable-http", host="0.0.0.0", port=a.port,
                   transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
