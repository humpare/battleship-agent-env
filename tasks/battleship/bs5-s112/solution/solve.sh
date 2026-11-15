#!/bin/bash
# Reference: exact elimination over every legal fleet, played through the same MCP tools.
set -euo pipefail
cd "$(dirname "$0")"
pip install --quiet --no-cache-dir numpy==1.26.4 >/dev/null 2>&1
export BATTLESHIP_N=5
python duel_client.py
