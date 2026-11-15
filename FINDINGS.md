# Findings

> **Superseded 2026-11-15 by `README.md`** (branch `battleship-v3`): the game is now called
> Battleship, the rules disclose the opponent, the evaluation uses 30 frozen boards, and the
> reference places by expected game score (0.994). The sections below are the record of the
> earlier iteration (five boards, tasks `tasks/duel5/`, kept in git tag `single-game-v1`).

Updated Saturday 2026-11-14, 08:15. Submission deadline: Sunday 2026-11-15, 10:00.
This file supersedes the 05:30 railway snapshot (kept below as section B).

## A. Battleship on a 5x5 board: the current headline environment

### The task
Classic Battleship (Battleship) as a real two-player game on a 5x5 board. Each side has five
ships (lengths 3, 2, 2, 1, 1), straight, never touching, not even diagonally. A hit shoots again,
a miss passes the turn. The agent places its own fleet, shoots first, and plays against a coded
opponent. Tools: `get_rules`, `observe`, `place_fleet`, `fire`. Tasks: `tasks/duel5/`.

### Why it is hard although it is tiny
Only 5,772 legal fleets exist. The no-touching rule forces ships to the edges: a corner cell
holds a ship in 60.8% of fleets, the four cells B2/D2/B4/D4 in 3.1%. Everything needed to work
that out is in the rules. A game is under 15 tool calls, so context length plays no role.

### The opponent and the reference (both deterministic, public information only)
- Opponent: lists every legal fleet, deletes those contradicted by the answers so far, shoots the
  cell with the best expected chain of hits. Ties are broken in a fixed random order per scenario,
  so it has no favourite cell and every run of a scenario is identical.
- Reference = the same exact search + reasoned placement: it places the fleet that costs that
  search method the most misses, averaged over tie-break orders (`scripts/placement_table.py`).

### Results through Harbor (judged harness), five boards (`results/final5-20261114/`)

| Run | s1 | s2 | s3 | s4 | s5 | Mean |
|---|---|---|---|---|---|---|
| Reference | 0.889 | 0.944 | 0.556 | 0.667 | 0.889 | 0.789 |
| Do nothing | 0 | 0 | 0 | 0 | 0 | 0.000 |
| Stop early (reference for 4 shots) | 0.167 | 0.222 | 0.056 | 0.056 | 0.056 | 0.111 |
| Kimi (kimi-k3-1-pro, OpenCode, MCP only) | 0.167 | 0.000 | 0.222 | 0.833 | 0.056 | 0.256 |

Kimi won 1 of 5. One run per board; a game takes Kimi about 3 minutes.

### Offline, 200 boards each, same opponent

| Player | Win rate | Mean score | Mean margin |
|---|---|---|---|
| Reference (exact search + reasoned placement) | 94% | 0.830 | +0.66 |
| Exact search + random legal placement | 62% | 0.557 | +0.11 |
| Exact search + a corner-heavy placement typical of the models | 48% | 0.463 | -0.07 |
| Checkerboard search + random placement | 26% | 0.346 | -0.31 |

So search skill is worth about 36 points of win rate and placement about another 32.

### Frontier models through the local arena (one-command client, not Harbor)
These were played before the opponent's tie-break was randomised (it then preferred A1 first).

| Model, board | Wins | Mean score | What happened |
|---|---|---|---|
| Opus 5.5, 5x5 edges, own placement | 1/5 | 0.144 | 8 of 30 shots on the 3.1% cells; fleet on corners |
| Fable 5.2, 5x5 edges, own placement | 2/5 | 0.533 | Never shot a 3.1% cell; counted layouts by hand; lost on corner placement |
| Opus 5.5, 6x6 wrap-around, own placement | 0/5 | 0.078 | NOT valid evidence: fleet anchored at A1, where the old tie-break started |
| Opus 5.5, 6x6 wrap-around, both fleets dealt (search only) | 2/5 | 0.411 | Near the exact algorithm in the same seat (3/5); pure search is not the weakness |

### What is established and what is not
- Established: the reference beats Kimi on the judged harness by a wide margin (0.789 vs 0.256),
  do-nothing scores 0, stop-early 0.111, results are deterministic for the coded runs.
- Established: the failure is reasoning, not overload, and is explainable cell by cell.
- Established: two separate weaknesses: (1) not deriving where ships can be on a crowded board,
  (2) not asking where a good searcher would look last when placing.
- Not established: three repeat runs per model on the final tasks; Opus and Fable through Harbor;
  a trained small model.
- Caveat: luck is large in single games; conclusions rest on averages.

### Score
Win: 0.5 + 0.5 x (own ship cells unhit / 9). Loss or unfinished: 0.5 x (enemy cells hit / 9).
Margin (reported next to it): (own cells afloat - enemy cells afloat) / 9, from -1 to +1.
The verifier replays the game server's private action log in a fresh game.

### Variants available as settings
Board size (`BATTLESHIP_N`), fleet (`BATTLESHIP_FLEET`), wrap-around board (`BATTLESHIP_WRAP=1`).
Exhaustive lists fit in memory up to 8x8 with five ships (121 million fleets, 0.9 GB).
Wrap-around removes the edge effect: every cell starts equal, the fleet no longer fits on 5x5,
and on 6x6 the exact algorithm's edge over the checkerboard player is 14 to 18 points
(against 32 to 37 on 5x5 with edges).

### Next stage (agreed direction, not started)
1. Three Kimi runs per board on the final tasks; Opus/Fable through Harbor if credits allow.
2. Small model: imitate the exact search and the reasoned placement, then measure it against the
   opponent and against the big models.
3. Series of games with learning of the opponent's habits; adversarial ("evil") small model.
4. README, report, demo.

## B. Railway dispatching (earlier environment, kept as supporting work)

Status at 05:30: works end to end, reward integrity holds, but no gap: Kimi matched the reference (0.989 vs 0.955).

Snapshot taken Saturday 2026-11-14, 05:30. Submission deadline: Sunday 2026-11-15, 10:00.
Raw artefacts for everything below are in `results/snapshot-20261114-0530/`.

## Status in one paragraph

The environment works end to end through Harbor (MCP-only agent, simulator in a sidecar,
verifier that replays a private action log). Reward integrity holds. **There is no gap against
a frontier model yet:** Kimi matches the reference on every task measured. The next design
target is a compact, hard-in-few-decisions task (about 15 decisions, so three agent runs fit a
10-minute demo) where a lookahead reference scores high and Kimi lands around 0.5 to 0.6.

## What is built

| Piece | File | State |
|---|---|---|
| Named-graph layout to Flatland 4.3.0 grid | `src/dispatch/layout.py`, `build.py` | Works; line with loops, long or short sidings |
| Simulator wrapper, commands, events, scoring, replay | `src/dispatch/sim.py` | Works; 24 tests, random-play fuzz |
| MCP server with 3 tools | `src/dispatch/server.py`, `service.py` | Works with OpenCode (mcp 2.3.0) |
| Reactive reference (safe hop-by-hop dispatcher) | `src/dispatch/reference.py` | 0 unfinished / 0 deadlocks on 1,500 random scenarios |
| Public forward model (rules + observation only) | `src/dispatch/model.py` | Exact vs simulator: 22,000+ decisions, with and without malfunctions |
| Lookahead reference (rollouts over priority rules) | `src/dispatch/lookahead.py` | Works offline; **not yet wired into the Harbor tasks** |
| Harbor task generator, 18 ladder tasks + 2 probes | `src/dispatch/gen_tasks.py`, `tasks/` | Works; oracle, nop, stop-early, OpenCode all run |
| Offline CLI: validate, run, compare, replay, branch | `src/dispatch/cli.py` | Works |
| Direct model runner (no Docker) | `src/dispatch/llm_runner.py` | Works; retries on rate limits |
| Harbor launchers | `scripts/evaluate.sh`, `harbor_verda.sh`, `exploit_checks.sh` | Work |
| Small-model training | `train/` | Dataset builder, LoRA trainer, server, Modal app written; **no trained model yet** |

## Measured results

### Harbor, held-out ladder (`harbor-table.txt`)

| Task | Trains | Reference (reactive) | Do-nothing | Stop-early | Kimi via OpenCode |
|---|---|---|---|---|---|
| a-basic-s101 / s102 | 4 | 1.000 / 1.000 | 0 / 0 | 0.250 / 0.500 | 1.000 / 1.000 |
| b-interaction-s101 / s102 | 6 | 1.000 / 1.000 | 0 / 0 | 0 / 0 | 1.000 / 1.000 |
| c-pressure-s101 / s102 | 6 | 0.938 / 1.000 | 0 / 0 | 0 / 0 | 1.000 / 1.000 |
| d-density-s101 / s102 | 10 | 1.000 / 0.932 | 0 / 0 | 0 / 0 | 1.000 / 0.910 |
| e-recovery-s101 / s102 | 10 | 0.967 / 0.837 | 0 / 0 | 0 / 0 | not finished |
| f-sustained-s101 / s102 | 16 | 0.960 / 0.825 | 0 / 0 | 0 / 0 | not finished |
| Mean | | 0.955 | 0.000 | 0.062 | 0.989 (8 tasks) |

Kimi = `kimi-k3-1-pro` on Verda, one run per task, MCP tools only. Reference scores are
identical in-process and through Harbor.

### How Kimi wins the easy stages

On c-pressure-s101 it beat the reactive reference (1.000 vs 0.938) with two command batches:
all trains of one direction get a full route as a convoy, the other direction is routed to a
loop and held, then released. It works because sidings hold 6+ trains, deadlines are loose and
there are only two directions.

### Kimi without OpenCode's reasoning setup

In the direct runner (same tools, plain tool-calling, no thinking configuration) the same model
scored 0.0 to 1.0 on stages a to d, mostly by giving opposing trains full routes and
deadlocking them, and once wrote "I've made a routing error". Valid runs:
a-basic 1.0 and 0.5, b-interaction 0.0 and 0.0, c-pressure 0.33 and 1.0, d-density 1.0 and 0.0,
e-recovery 0.1 and 0.3. The first sweep in `invalid/` was cut short by HTTP 429 and must not
be used.

### Local model

`qwen3.8:27b` through Harbor + OpenCode needs minutes per step. In its one (stopped) run it gave
every train a full end-to-end route on the first step.

### Reward integrity (`exploit_checks.log`)

Three cheating "agents" run as root in the agent container all scored 0: forging
`/logs/verifier/reward.txt` and a fake action log; hammering the server with malformed and
illegal requests; holding every train forever.

### Reward cross-check against Flatland's own reward

Same 56 episodes scored with Flatland 4.3.0 `DefaultRewards.normalize`:

| Policy | Ours | Flatland normalised | Completion | On time |
|---|---|---|---|---|
| Kimi (8) | 0.989 | 0.998 | 100% | 93% |
| Reactive reference (12) | 0.956 | 0.993 | 100% | 73% |
| One train at a time (12) | 0.662 | 0.740 | 100% | 25% |
| Full routes for all (12) | 0.000 | 0.207 | 0% | 0% |
| Do nothing (12) | 0.000 | 0.881 | 0% | 0% |

Same ranking for policies that play. Flatland's reward gives do-nothing 0.88, which would fail
the hackathon's "doing nothing earns little" test; ours gives 0. Weakness of ours: under heavy
congestion everything saturates near 0.5.

### Lookahead reference, offline

| Setting | Reactive | Lookahead | Lookahead on-time |
|---|---|---|---|
| d-density (3 seeds) | 0.955 | 0.997 | 87% |
| f-sustained | 0.903 | 0.946 | 56% |
| g-scarce (14 trains, sidings hold 4) | 0.625 | 0.705 | 17% |
| h-heavy (20 trains, 4 loops) | 0.626 | 0.638 | 5% |

g and h as configured are overloaded: no dispatcher can be on time there.

### Published Flatland baseline

`flatland-baselines` deadlock-avoidance policy (commit c1733cb) reads the simulator object
directly, so it cannot be the same-information reference. Run directly on our layouts it stops
on an assertion during path setup in all 18 scenarios; not debugged
(`scripts/published_baseline.py`).

## Constraints discovered

- **Demo must fit 10 minutes.** Kimi takes about 15 s per step through OpenCode. A 26-step task
  took about 8 minutes; a 100-step task would take 25 to 40. Target: about 15 decisions per episode.
- **Token cost grows with the square of episode length** (the agent re-reads the conversation
  each step). Rough estimates: 1.5 to 2 M input tokens for a 40-step run, 12 to 16 M for 100 steps.
- **Verda rate limits:** 9 parallel runs triggered HTTP 429; 2 to 3 in parallel worked.
- **Modal:** every GPU type is refused until a payment method is added to the account.
  `train/modal_train.py` is ready.
- **Local GPU (24 GB):** LoRA on Qwen2.5-3B needs gradient checkpointing; about 95 minutes for
  7,000 examples. The one real run was stopped at 47% with nothing saved.
- **Never delete or regenerate a task directory while a Harbor run on it is in flight:** the
  verifier image is built from `tests/` at the end.
- **MCP-only restriction** is one `ENV OPENCODE_CONFIG_CONTENT` line in the task Dockerfile; in
  Harbor runs Kimi used only the three `dispatch_*` tools. Not yet confirmed with the organisers.

## Agreed next steps (not started)

1. Compact Y-junction layout with small sidings, 8 to 10 trains, about 15 decisions:
   stop the clock only on events that need the dispatcher.
2. Slim the observation (drop static per-train fields).
3. Wire the lookahead reference into the Harbor tasks.
4. Offline filter: keep a setting only if lookahead scores about 0.85+ and a coded "convoy"
   strategy does not.
5. Kimi on the survivors; tune toward 0.5 to 0.6; confirm with three runs.
6. Per-step projected reward in the private log; widen the lateness term if it saturates.
7. Restart the small-model fine-tune; before/after comparison.
8. README, report, final reruns on frozen code.

## How to check the state of the repo

    uv run pytest -q                       # 24 tests
    uv run python -m dispatch.cli validate # every scenario feasible and deterministic
    uv run python tests/fuzz_model.py 200  # forward model == simulator
    scripts/evaluate.sh                    # Harbor baselines on tasks/heldout
