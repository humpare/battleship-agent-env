#!/bin/bash
# Run the Verda-hosted model as the OpenCode agent through Harbor.
#   scripts/harbor_verda.sh TASKS JOB_NAME [ATTEMPTS] [PARALLEL] [MODEL]
# Needs VERDA_API_KEY in the environment or in .env (never written to the job directory).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
TASKS=$1; JOB=$2; K=${3:-1}; N=${4:-2}; MODEL=${5:-kimi-k3-1-pro}
CFG='{"provider":{"verda":{"npm":"@ai-sdk/openai-compatible","options":{"baseURL":"https://inference.verda.com/v1","apiKey":"{env:VERDA_API_KEY}"},"models":{"'$MODEL'":{}}}}}'
uv run harbor run -p "$TASKS" -a opencode -m "verda/$MODEL" --api-key-name VERDA_API_KEY \
    --ak "opencode_config=$CFG" -k "$K" -n "$N" -o runs/jobs --job-name "$JOB"
