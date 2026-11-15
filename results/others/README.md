# Other scenarios we analysed

Everything measured outside the judged five-board evaluation. Game logs and Harbor outputs are
kept as recorded, with local paths shortened. "Earlier version" means before the rules disclosed
the opponent and before the reference placed by expected game score; those numbers are not
comparable with the main README table.

## Same game, other conditions for the agent (current version, through Harbor)

| Folder | Scenario | Result |
|---|---|---|
| `kimi-code-allowed-20261115/` | Kimi may also run code in its container (10 boards x 3 runs). It wrote its own fleet counter and chose a fleet with empty corners. | mean 0.959, 29 wins of 30, shot quality 0.989 |
| `kimi-strategy-advice-20261115/` | MCP tools only, plus three paragraphs of strategy advice in the instruction (10 boards; 18 valid runs, 12 lost to a provider key error) | mean 0.861, 16 wins of 18, shot quality 0.959 |

Each has one complete Harbor trial with the agent's trajectory (`full-harbor-log-*`). Read
together with the judged result (0.467 with MCP tools only): the task is solvable from the
rules, and the agent's failure is not deriving the answer unprompted.

## Earlier version of the game

| Folder | Scenario | Result |
|---|---|---|
| `final5-20261114/` | Harbor, five boards, 5x5: reference, do-nothing, stop-early, Kimi (one run per board) | reference 0.789, do-nothing 0.000, stop-early 0.111, Kimi 0.256 (1 win of 5) |
| `opus-vs-exact-5x5-20261114/` | Claude Opus 5.5, five single games, local harness | 1 win of 5, mean 0.144; 8 of 30 shots at the 3.1% cells |
| `fable-vs-exact-5x5-20261114/` | Claude Fable 5.2, five single games, local harness | 2 wins of 5, mean 0.533; counted fleets by hand, lost on corner placement |
| `opus-search-only-6x6wrap-20261114/` | Opus 5.5 on a 6x6 wrap-around board with both fleets dealt (search only) | 2 wins of 5, mean 0.411; the exact player in the same seat won 3 of 5 |
| `opus-vs-exact-6x6wrap-20261114/` | Opus 5.5 on 6x6 wrap-around, own placement | 0 of 5; NOT valid evidence (the opponent's old tie-break favoured the cells Opus used) |
| `opus-series-5x5-20261114/` | Three series of five games against the first adaptive opponent, with the history visible | Opus lost every first game, then won 10 of the next 12 |
| `opus-series-v2-5x5-20261114/` | One series of five against the improved adaptive opponent | Opus won 1 of 5 |
| `duel-poc-20261114/`, `seafight-poc-20261114/` | First prototypes (7x7 duel; one-sided 10x10 game) | development records |

Model quotes and the check of their hand counts against the exact count: `docs/OBSERVATIONS.md`.

## Railway dispatching (the first environment we built)

| Folder | Scenario | Result |
|---|---|---|
| `snapshot-20261114-0530/` | Trains on a single-track line with passing loops (Flatland), Harbor, held-out tasks | reference 0.955, Kimi 0.989: no gap, so we changed the task; exploit checks all 0 |

## Simulations with coded players only

`simulations/`: cell chances, value of every fleet, the reference against itself, counts of legal
fleets for other boards and fleets, exact search against simpler players on 5x5 to 8x8 and
wrap-around boards, and series against an adapting opponent.

## Small model

`results/small-model/` (one level up): a 3B model before and after training, and the
mixed-rules transfer test.
