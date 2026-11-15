#!/bin/bash
# Evaluate a local small model (optionally with a LoRA adapter) as a stateless dispatcher.
#   train/eval_model.sh NAME [ADAPTER_DIR] [SCENARIO_GLOB]
# e.g. train/eval_model.sh base
#      train/eval_model.sh bc-v1 train/out/bc-v1 "scenarios/heldout/*.json"
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1; ADAPTER=${2:-}; GLOB=${3:-scenarios/heldout/*.json}; PORT=${PORT:-8011}
train/.venv/bin/python train/serve.py --port "$PORT" ${ADAPTER:+--adapter "$ADAPTER"} > "runs/serve-$NAME.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 120); do grep -q "^serving" "runs/serve-$NAME.log" 2>/dev/null && break; sleep 2; done
# shellcheck disable=SC2086
uv run python -W ignore -m dispatch.llm_runner --mode stateless --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$NAME" --max-tokens 512 --out "runs/llm/$NAME" --scenario $GLOB | tee "runs/llm-$NAME.jsonl"
