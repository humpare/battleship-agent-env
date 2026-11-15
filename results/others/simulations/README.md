# Simulations (coded players only, no language model)

Files here were produced on 2026-11-15 with the current code. The tables below marked
"recorded 2026-11-14" are from the earlier session; their raw outputs were printed, not saved,
and they used the earlier version of the game (enemy fleet drawn ship by ship, reference placing
by the older table). The scripts that produce them are named so they can be rerun.

## Current version, 5x5

- `cell-chances-5x5.txt`: share of the 5,772 legal fleets with a ship in each cell (corners 60.8%,
  centre 52.3%, B2/D2/B4/D4 3.1%).
- `coded-players-30-boards.txt` (`scripts/offline_eval.py`): reference 0.994 (30 wins), exact
  search with a random fleet 0.663 (22), checkerboard hunt 0.356 (7), random shots 0.115 (0),
  stop after four shots 0.156.
- `fleet-value-table.txt` (`scripts/placement_duel.py`): expected game score of every legal
  fleet against the disclosed opponent: best 0.99, mean 0.59, worst 0.24.
- Reference against itself (`scripts/mirror_check.py`, 400 games, both sides placing by the
  reference's rule): the side that shoots first wins 72.8%.

## How many legal fleets there are (computed 2026-11-15 and 2026-11-14)

| Board | Fleet | Legal fleets |
|---|---|---|
| 5x5 | 3-2-2-1-1 | 5,772 |
| 5x5 | 3-2-1-1 | 8,712 |
| 5x5 | 2-2-2-1 | 6,144 |
| 5x5 | 3-3-1 | 1,026 |
| 5x5 | 3-2-2-1 | 6,508 |
| 5x5 | 2-2-1-1-1 | 10,016 |
| 6x6 | 3-2-2-1-1 | 618,552 |
| 6x6 | 3-3-2-1 | 89,968 |
| 6x6 | 4-2-2-1-1 | 258,428 |
| 6x6 | 2-2-2-1-1 | 447,852 |
| 6x6 wrap-around | 3-2-2-1-1 | 112,968 |
| 7x7 | 3-2-2-1-1 | 12.9 million |
| 8x8 | 3-2-2-1-1 | 121 million (wrap-around: 116 million); 0.9 GB, the largest list that fits a normal container |
| 8x8 | six ships | 749 million to 1.13 billion |
| 8x8 | seven ships | 4.2 billion |

The five-ship fleet does not fit on a 5x5 wrap-around board.

## Board size: exact search against the checkerboard player (recorded 2026-11-14, `scripts/sim_matrix.py`)

Win rate of each player against the same exact opponent, random legal placement:

| Board | Exact search | Checkerboard hunt |
|---|---|---|
| 5x5 | 84% | 52% |
| 6x6 | 68% | 49% |
| 6x6 wrap-around | 70% | 56% |
| 7x7 | 72% | 54% |
| 8x8 | 63% | 50% |

The exact player's edge shrinks as the board grows, and wrap-around removes the corner
structure (every cell starts equal). This is why the judged tasks use 5x5 with edges.

## What search and placement are each worth (recorded 2026-11-14, 200 boards, earlier version)

| Player | Win rate | Mean score |
|---|---|---|
| Reference (exact search + placement table) | 94% | 0.830 |
| Exact search + random legal fleet | 62% | 0.557 |
| Exact search + a corner-heavy fleet typical of the models | 48% | 0.463 |
| Checkerboard hunt + random fleet | 26% | 0.346 |

If the opponent also chooses its fleet well: reference 65% (0.622), corner-style player about 7% (0.161).

## Series of five games against an opponent that adapts (recorded 2026-11-14, `scripts/series_duel.py`, `scripts/series_placement.py`)

A coded stand-in for a learning player (it copies what worked in the previous game) won 50% of
games 2 to 5 against the first adaptive opponent and 7% against the improved one (which learns
search weights from past fleets, shoots cells the player used before, and never repeats a fleet).
The games Claude Opus played in the same two settings are in `../opus-series-5x5-20261114/` and
`../opus-series-v2-5x5-20261114/`.
