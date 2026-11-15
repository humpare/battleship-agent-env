#!/bin/bash
# Coded reference dispatcher. Plays through the same three MCP tools the agent gets.
set -euo pipefail
cd "$(dirname "$0")"
# DISPATCH_POLICY=nothing and DISPATCH_MAX_STEPS=N turn it into the do-nothing / stop-early baselines.
python mcp_client.py --url http://sim:8000/mcp --policy "${DISPATCH_POLICY:-reference}" \
    ${DISPATCH_MAX_STEPS:+--max-steps "$DISPATCH_MAX_STEPS"}
