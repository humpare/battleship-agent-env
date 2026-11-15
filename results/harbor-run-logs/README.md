# Harbor run logs

Two complete trials of `kimi-k3-1-pro` driving OpenCode through Harbor on the judged tasks
(`harbor run -p tasks/battleship5 -a opencode -m verda/kimi-k3-1-pro -k 3`), copied from the job
directory with local paths shortened:

| Trial | Board | Score (`verifier/reward.txt`) |
|---|---|---|
| `bs5-s105__*` | bs5-s105 | 1.000000 (won with its own fleet untouched) |
| `bs5-s102__*` | bs5-s102 | 0.222222 (lost; hit 4 of 9 enemy cells) |

Each folder has Harbor's `result.json` and `trial.log`, the agent's `trajectory.json` and raw
`opencode.txt`, the verifier's `reward.txt` and `result.json`, and the game server's private
action log `artifacts/data/episode.jsonl` that the verifier replays. `job-result.json` is the
summary of the whole 15-trial job. Scores and logs of all 15 trials are in
`results/battleship-v3/v4-kimi-mcp5/`.
