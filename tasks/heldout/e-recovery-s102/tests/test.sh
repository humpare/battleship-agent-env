#!/bin/bash
# Scores the episode by replaying the simulator sidecar's private action log.
mkdir -p /logs/verifier
python /tests/verify.py --scenario /tests/scenario.json --log /data/episode.jsonl --out /logs/verifier
