#!/bin/bash
# One-command evaluation through Harbor: reference, do-nothing, stop-early and (optionally)
# (railway tasks: pass -t tasks/heldout -s 8)
# an OpenCode agent, on a task or a directory of tasks. Prints one table at the end.
#
#   scripts/evaluate.sh                                   # baselines on the 5 boards of tasks/battleship5 (use -t tasks/battleship for all 30)
#   scripts/evaluate.sh -m anthropic/claude-opus-4-5      # + 3 agent attempts per task
#   scripts/evaluate.sh -t tasks/heldout/d-density-s101 -m openai/gpt-5 -k 3
#
# Options: -t TASKS (task dir or directory of tasks)   -m MODEL (provider/model for OpenCode)
#          -k ATTEMPTS (agent attempts per task, default 3)   -n PARALLEL (default 4)
#          -s STEPS (stop-early baseline stops after this many reference steps, default 4)
#          -x "ARGS" (extra arguments passed to the agent's `harbor run`, e.g. --ak ...)
set -euo pipefail
cd "$(dirname "$0")/.."
TASKS=tasks/battleship5; MODEL=""; K=3; N=5; STEPS=4; EXTRA=""
while getopts "t:m:k:n:s:x:" o; do
  case $o in t) TASKS=$OPTARG;; m) MODEL=$OPTARG;; k) K=$OPTARG;; n) N=$OPTARG;; s) STEPS=$OPTARG;; x) EXTRA=$OPTARG;; *) exit 2;; esac
done
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=runs/jobs
mkdir -p "$OUT"
run() { echo ">> $1"; shift; uv run harbor run -p "$TASKS" -o "$OUT" "$@" > "$OUT/$STAMP.$JOB.log" 2>&1 || echo "   (harbor exited non-zero, see $OUT/$STAMP.$JOB.log)"; }

JOB=reference;  run "reference (coded player via the same MCP tools)" -a oracle -n "$N" --job-name "$STAMP-reference"
JOB=nothing;    run "do-nothing baseline"                                 -a nop    -n "$N" --job-name "$STAMP-nothing"
JOB=stop-early; run "stop-early baseline (reference for $STEPS steps, then stops)" -a oracle --ae DISPATCH_MAX_STEPS="$STEPS" --ae BATTLESHIP_MAX_SHOTS="$STEPS" -n "$N" --job-name "$STAMP-stop-early"
COLS=("reference=$OUT/$STAMP-reference" "nothing=$OUT/$STAMP-nothing" "stop-early=$OUT/$STAMP-stop-early")
if [ -n "$MODEL" ]; then
  JOB=agent
  # shellcheck disable=SC2086
  run "OpenCode agent $MODEL, $K attempts per task" -a opencode -m "$MODEL" -k "$K" -n "$N" --job-name "$STAMP-agent" $EXTRA
  COLS+=("agent=$OUT/$STAMP-agent")
fi
echo
uv run python -m dispatch.report "${COLS[@]}" --json "$OUT/$STAMP-summary.json"
echo
echo "Details: $OUT/$STAMP-*/<trial>/verifier/result.json   Summary: $OUT/$STAMP-summary.json"
