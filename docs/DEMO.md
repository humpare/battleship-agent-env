# Ten-minute demo plan

Before the slot: Docker running, `uv sync` done, images built once (run `scripts/evaluate.sh`
beforehand so the layers are cached), provider key in `.env`.

| Minute | Show | Command or file |
|---|---|---|
| 0 to 1 | The task in one breath: 5x5 Battleship, four MCP tools, opponent's method disclosed, score replayed from a private log | `README.md`, "The task" |
| 1 to 2 | Start the judged evaluation: reference, do-nothing, stop-early, then three agent runs on five boards | `scripts/evaluate.sh -m PROVIDER/MODEL -k 3` (or `scripts/harbor_verda.sh tasks/battleship5 demo 3 5`) |
| 2 to 4 | While it runs: why the game is hard although tiny. Corners 60.8%, B2/D2/B4/D4 3.1%; fleet value from 0.24 to 0.99 | `README.md`, "Two decisions decide the game" |
| 4 to 6 | A recorded agent game, shot by shot, with the grade of each shot and the fleet it chose | `uv run python scripts/bs_report.py ref=... agent=...`, a trial's `episode.jsonl` |
| 6 to 7 | Reward integrity: tests and the four cheating scripts | `uv run pytest -q`, `scripts/exploit_checks.sh` (recorded output) |
| 7 to 8 | A small model improves with training: 0.000 to 0.811 | `README.md`, small-model tables |
| 8 to 10 | Back to the live table: reference near 1.0, do-nothing 0, stop-early about 0.15, agent runs | terminal |

Timing measured on this machine: the three coded baselines on five boards take about two
minutes with cached images; one agent game takes about three minutes; five boards run in parallel.
