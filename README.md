# Battleship: exact belief tracking against a disclosed opponent

Entry for AfterQuery's "Break the Agent" challenge. The agent plays two-player Battleship on a
5x5 board against a coded opponent, through four MCP tools. The game is tiny (5,772 legal
fleets, about 15 tool calls), every rule and the opponent's method are given to the agent, and
the right move at every step can be computed exactly. A coded reference wins almost every
game; a frontier agent playing through the same tools does not.

> 🔴 **Trained models available on request.** Both fine-tuned models (Qwen2.5-3B LoRA adapters: the
> standard-rules model that goes from 0.000 to 0.811, and the mixed-rules model) are uploaded to
> IronShard object storage and ready for review. Ask us and we will share access.

Demo page: https://junction.danielegerev.com
Game replays with exact hit-chance heat maps: https://junction.danielegerev.com/replay.html
(source in `site/`, data built by `scripts/make_replays.py`)

## At a glance

| | Mean score | Wins |
|---|---|---|
| Reference (exact calculation) | **0.989** | 5 of 5 |
| Frontier agent (Kimi K3.1 Pro, MCP tools only) | **0.467** | 5 of 15 |
| Do nothing | **0.000** | 0 of 5 |
| Stop after 4 shots | **0.144** | 0 of 5 |
| Small model (Qwen2.5-3B) before and after training | **0.000 → 0.811** | 0 → 24 of 30 |

- **To win, calculate.** At every shot the best move is the cell most likely to hold a ship given
  what has been seen so far, and that chance can be computed exactly.
- **The opponent calculates everything that can be calculated**, so each imprecise shot or a
  predictable fleet costs the game or most of the score.
- **The agent plays from habit instead**: it opens on the centre and puts ships on the corners.

## Run the evaluation (one command)

```bash
uv sync                                        # Python 3.12, Docker running
scripts/evaluate.sh                            # reference, do-nothing, stop-early on the 5 boards
scripts/evaluate.sh -m PROVIDER/MODEL -k 3     # plus three agent runs per board (OpenCode via Harbor)
```

One iteration is five games (`tasks/battleship5/`, boards 101 to 105). The coded runs take about
two minutes; an agent game takes about three minutes, and the boards run in parallel.
`scripts/evaluate.sh -t tasks/battleship` runs all 30 frozen boards, and
`-t tasks/battleship5/bs5-s101` a single one. For an OpenAI-compatible provider that OpenCode
does not know, see `scripts/harbor_verda.sh` (it passes the provider settings to OpenCode).

| Command | What it does |
|---|---|
| `uv run pytest -q` | 68 tests, including the reward-integrity tests (no Docker needed) |
| `scripts/exploit_checks.sh` | four cheating scripts through Harbor; none may gain reward |
| `uv run python scripts/offline_eval.py` | coded players on the 30 boards without Docker (seconds) |
| `uv run python scripts/mirror_check.py` | the reference against itself |
| `uv run python scripts/bs_report.py ref=runs/jobs/A agent=runs/jobs/B` | scores with intervals and a grade for every shot |
| `train/bs_pipeline.sh` | small-model experiment on a 24 GB GPU |

## The task

- **Board and fleets.** 5x5. Each side hides five ships of lengths 3, 2, 2, 1, 1. Ships are
  straight and never touch, not even diagonally.
- **Turns.** The agent places its fleet, then shoots first. A hit shoots again; a miss passes
  the turn. First to sink the other fleet wins.
- **Tools.** `get_rules`, `observe`, `place_fleet`, `fire`. Nothing else: the agent is a player,
  not a programmer.
- **The opponent is disclosed.** `get_rules` states that its fleet is drawn uniformly from all
  legal fleets, and that it shoots by listing every fleet consistent with its results and taking
  the most likely cell. Nothing the agent needs is withheld.
- **Score.** A win gives 0.5 plus 0.5 times the share of the agent's own ship cells still unhit.
  A loss or an unfinished game gives 0.5 times the share of enemy ship cells hit.
- **Tasks.** `tasks/battleship/` holds 30 frozen boards (seeds 101 to 130) and `tasks/battleship5/`
  the first five of them, in Harbor format:
  game server in a sidecar container, verifier in a separate container without network.

### Two decisions decide the game

1. **Where to shoot.** The no-touching rule pushes ships to the rim. A corner holds a ship in
   60.8% of legal fleets, the centre in 52.3%, and B2, D2, B4, D4 in 3.1%. A diagonal or
   checkerboard hunt, the usual habit, spends shots on the 3.1% cells.
2. **Where to place.** The opponent shoots likely cells first, so a fleet that leaves the corners
   empty makes its opening shots miss. Over all 5,772 fleets, the expected score of an exact
   shooter ranges from 0.24 (worst fleet) to 0.99 (best fleet), with a mean of 0.59.

Both follow from the rules text. Neither needs information the agent does not have.

Share of the 5,772 legal fleets with a ship in each cell (%):

| | A | B | C | D | E |
|---|---|---|---|---|---|
| **1** | **60.8** | 35.5 | 54.0 | 35.5 | **60.8** |
| **2** | 35.5 | 3.1 | 23.0 | 3.1 | 35.5 |
| **3** | 54.0 | 23.0 | 52.3 | 23.0 | 54.0 |
| **4** | 35.5 | 3.1 | 23.0 | 3.1 | 35.5 |
| **5** | **60.8** | 35.5 | 54.0 | 35.5 | **60.8** |

This is the opening position, before any shot. After each result the list of possible fleets
shrinks and the numbers change; the reference recomputes them before every shot.


## Results

Through Harbor, five boards (`results/battleship-v3/`; two complete Harbor run logs with their
scores are in `results/harbor-run-logs/`). The agent is `kimi-k3-1-pro` driving
OpenCode with the game's MCP tools only, three runs per board.

| Run | s101 | s102 | s103 | s104 | s105 | Mean | Wins |
|---|---|---|---|---|---|---|---|
| Reference | 1.000 | 1.000 | 1.000 | 0.944 | 1.000 | 0.989 | 5/5 |
| Do nothing | 0 | 0 | 0 | 0 | 0 | 0.000 | 0/5 |
| Stop after 4 shots | 0.167 | 0.167 | 0.111 | 0.111 | 0.167 | 0.144 | 0/5 |
| Kimi, three runs per board | 0.28 / 0.72 / 0.28 | 0.28 / 0.44 / 0.22 | 0.72 / 0.22 / 0.44 | 0.44 / 0.56 / 0.56 | 0.39 / 1.00 / 0.44 | 0.467 | 5/15 |

Kimi's shot quality is 0.91 over all shots (0.869 while more than 100 fleets
are still possible, 0.945 after that); the reference's is 1.000.

**The gap:** the reference is 0.522 ahead of the agent on average (95% interval 0.433 to 0.619, over
boards), and ahead on every board. The agent won 5 of its 15 games; the reference won all 5.

- **Shot quality** is the chosen cell's chance of a hit divided by the best available chance,
  from the exact count of remaining fleets. A random shooter scores 0.49, a checkerboard hunter 0.76.
- **All 30 frozen boards, coded runs:** reference 0.994 (30 wins), do-nothing 0.000, stop after
  four shots 0.156.
- **Coded yardsticks on the 30 boards:** exact search with a random legal fleet 0.663 (22 wins);
  checkerboard hunt with a random fleet 0.356 (7 wins); random shots 0.115 (0 wins). So both
  decisions carry weight: search and placement.
- **The reference against itself** (both sides search exactly and place by the same rule, 400
  games): the side that shoots first wins 72.8%. The first-move edge is large because a hit
  shoots again and both sides usually need the same small number of turns; the player who
  starts wins those ties.

## Why the agent fails

Every shot and every fleet in the agent's games is graded against the exact count
(`scripts/bs_report.py`). In these 15 games:

- **Placement is where most of the score is lost.** 15 of 15 fleets put a ship on a corner, the
  cells the rules say the opponent shoots first. The reference keeps all four corners empty and
  finishes with 8.8 of its 9 ship cells unhit on average; Kimi finishes with 1.3.
  The rules state how the opponent chooses its shots, so this follows from the task text.
- **Shooting is good but not exact.** 44% of Kimi's shots were not at a best cell, and
  0% went to cells with under a 5% chance of a hit. Each miss hands the opponent a turn.
- **Its first shot was C3 in 14 of 15 games.**

In an earlier version of the task, Kimi and Claude Opus 5.5 hunted on the diagonal, where a
ship sits in 3.1% of fleets, yet counted the remaining fleets exactly by hand once fewer than
about 100 were left (`docs/OBSERVATIONS.md`). The failure is a decision, not a missing
capability: whether to play the familiar game or to work out what these rules and this
opponent imply.

## Reward integrity

- **The score is recomputed, not reported.** The game server writes a private action log. The
  verifier, in its own container with no network, replays the log against the scenario and
  writes the reward. The agent's container holds no game code and no scenario.
- **Doing nothing scores 0; placing a fleet and stopping scores 0.**
- **Stopping early never pays.** The score never falls as the agent plays on, and any win
  (at least 0.556) beats any unfinished game (at most 0.444). Tested on random games.
- **Illegal, repeated and simultaneous requests change nothing.** Sixty parallel shots at one
  cell count once.
- **Nothing leaks.** Answers never contain an unshot enemy cell or the score.
- **The opponent uses public information only.** A test rebuilds every opponent shot from what
  an onlooker sees of the agent's board.
- **Harbor-level checks** (`scripts/exploit_checks.sh`, run as root with a shell in the agent
  container, which is more than the judged agent has): forged reward files and a forged log,
  junk requests, place-and-stop, and a flood of shots. All four leave the reward at 0.

A limit to state plainly: the opponent is fixed and disclosed, so one fleet is close to best
against it on every board. An agent that knows that fleet in advance and searches exactly
scores about 0.99. This is the intended solution, not an exploit, but it means the task
measures whether the agent derives it, not whether it can be surprised.

## A smaller model improves with training

Qwen2.5-3B-Instruct, LoRA, trained to imitate the exact player's decisions (supervised
imitation, not reinforcement learning). It plays through the real game service, one short chat
turn per decision, on boards never used in training. Scores are strict: a game in which the
model never places a legal fleet scores 0.

**Learning curve, standard rules** (`train/bs_pipeline.sh`, 55,060 decisions from 4,000 games):

| Training steps | 30 task boards | Wins | 100 further boards | Wins | Shot quality |
|---|---|---|---|---|---|
| 0 (untrained) | 0.000 | 0 | 0.000 | 0 | 0.38 |
| 200 | 0.000 | 0 | 0.000 | 0 | |
| 800 | 0.000 | 0 | 0.109 | 0 | |
| 2,400 | 0.417 | 6 | 0.377 | 16 | 0.89 |
| 6,883 (one pass) | **0.811** | **24** | **0.746** | **63** | 0.93 |

The untrained model cannot place a legal fleet and re-shoots cells it has already shot.

**One model trained on seven rule sets, tested on rules it never saw** (`train/modal_bs.py`,
141,382 decisions; 30 boards per rule set):

| Rule set | In training | Untrained | Trained | Trained, fleet dealt to it | Shot quality, trained |
|---|---|---|---|---|---|
| 5x5, fleet 3-2-2-1-1 | yes | 0.000 | 0.789 (22 wins) | 0.789 | 0.93 |
| 5x5, fleet 3-2-1-1 | yes | 0.000 | 0.552 (19 wins) | 0.552 | 0.96 |
| 5x5, fleet 3-2-2-1 | no | 0.000 | 0.000 | 0.460 | 0.94 |
| 5x5, fleet 2-2-1-1-1 | no | 0.000 | 0.000 | 0.533 | 0.93 |
| 6x6, fleet 3-2-2-1-1 | no | 0.000 | 0.420 (8 wins) | 0.420 | 0.93 |

- **Shooting carries over to unseen rules** (shot quality 0.93 to 0.94 against 0.43 to 0.49 untrained).
- **Placement does not.** On the two unseen fleets the model never produced a legal fleet, so the
  strict score is 0. A first training round placed legally on one of them; this part is fragile.
- Results: `results/small-model/` (the first round, trained on weaker placement examples, is
  in `v1-old-placement/`: 0.706 on the 30 task boards).

## What is new here and what is not

- **Not new:** probability-count Battleship shooting; language models playing Battleship
  (Grand et al., "Loose LIPS Sink Ships", 2024, and "Shoot First, Ask Questions Later?", 2025;
  TextArena includes a Battleship environment); language models choosing predictably.
- **What this adds:** a replay-verified, fully enumerable version in which every shot and every
  placement can be graded exactly, and a
  small model that learns the task and carries part of it to unseen rules.
- **The reference is not proven optimal.** It eliminates fleets exactly, shoots for the longest
  expected run of hits over three shots, and places the fleet with the best expected score
  against the disclosed opponent.

## Limits

- **One agent model** (Kimi) in the Harbor study: 15 games on five boards.
- **A fixed, disclosed opponent** makes one fleet near-best on every board. A series against an
  opponent that adapts between games exists (`src/battleship/adaptive.py`, `arena/`) but is not
  part of the judged tasks.
- **The small model learned by imitation**, and its placement does not transfer to new fleets.

## Repository map

| Path | Content |
|---|---|
| `src/battleship/` | game, opponent, reference player, MCP server, task generator |
| `tasks/battleship5/`, `tasks/battleship/` | the five-game iteration and the full set of 30 frozen boards |
| `tests/test_battleship.py` | integrity and rules tests |
| `scripts/` | evaluation, exploit checks, reports, placement tables |
| `train/` | small-model data, training and evaluation (local and Modal) |
| `results/` | recorded runs: `battleship-v3/` (judged evaluation), `harbor-run-logs/`, `small-model/`, and `others/` (every other scenario we analysed, with an index); `docs/OBSERVATIONS.md` has model quotes |
| `src/dispatch/`, `tasks/heldout/` | an earlier railway-dispatching environment in which the frontier agent matched the reference (0.989 against 0.955); kept as supporting work, see `FINDINGS.md` section B |
