## What we built

A reinforcement learning environment in the Harbor task format: two-player Battleship on a 5x5 board, played by the agent through four MCP tools (`get_rules`, `observe`, `place_fleet`, `fire`). Each side hides five ships (lengths 3, 2, 2, 1, 1) that may not touch, not even diagonally. A hit shoots again; a miss hands the turn to the opponent.

The game is tiny: 5,772 legal fleets and about 15 tool calls. Every rule, and the opponent's exact method, is given to the agent. Nothing is hidden except the two fleets.

## Why it is hard

**The way to win is to calculate.** At every shot the right move is the cell most likely to hold a ship, given everything seen so far. That probability can be computed exactly by listing the fleets still possible. The same holds for placing your own fleet: the opponent shoots the likeliest cells first, so a good fleet stays out of them.

**The opponent calculates everything that can be calculated.** It is a program that lists every fleet consistent with its results and shoots the most likely cell, every time. It never guesses from habit and never tires.

**So one imprecise decision is close to fatal.** Each miss gives that opponent a turn, and a predictable fleet is found in a few shots. Our reference player, which makes both decisions by exact calculation, wins every game. A player who shoots exactly but places carelessly drops from 0.99 to 0.66.

Two facts follow from the rules and decide most games:

- A corner holds a ship in 60.8% of legal fleets, the centre in 52.3%, and the four cells diagonally next to a corner in 3.1%.
- The value of a fleet ranges from 0.24 (worst) to 0.99 (best) against this opponent. The best fleets leave all four corners empty.

## Results (Harbor, OpenCode harness, five boards)

| Run | Mean score | Wins |
|---|---|---|
| Reference (exact calculation) | 0.989 | 5 of 5 |
| Do nothing | 0.000 | 0 of 5 |
| Stop after 4 shots | 0.144 | 0 of 5 |
| Frontier agent (Kimi K3.1 Pro), 3 runs per board | 0.467 | 5 of 15 |

The reference is 0.52 ahead on average (95% interval 0.43 to 0.62) and ahead on every board. On all 30 frozen boards the reference scores 0.994.

## Why the agent fails

We grade every shot and every fleet against the exact count.

- **Placement:** in 15 of 15 games the agent put a ship on a corner, the cells the rules say the opponent shoots first. It finished with 1.3 of its 9 ship cells unhit on average; the reference finished with 8.8.
- **Shooting:** 44% of its shots were not at a best cell. It opened on the centre in 14 of 15 games.

Large language models carry patterns of play from the huge body of text they were trained on, and they apply those patterns here: open in the centre, anchor ships in corners. They were not trained on this exact game, so they play the familiar game instead of calculating this one, and against an opponent that calculates everything, that is enough to lose. The failure is a decision, not missing information: in earlier games the same models counted the remaining fleets correctly by hand once few were left.

## Reward integrity

- The score is recomputed, not reported: the game server keeps a private action log, and a separate verifier container without network replays it and writes one number between 0 and 1 to `/logs/verifier/reward.txt`.
- Partial credit is built in: a loss scores half the share of enemy cells hit; a win scores 0.5 plus half the share of your own cells still unhit.
- Doing nothing scores 0. Stopping early never beats playing on: the score never falls, and any win beats any unfinished game.
- 68 tests cover replay, illegal and repeated requests, simultaneous shots and leaks. Four cheating scripts run through Harbor (forged reward files, junk requests, place-and-stop, a flood of shots) all scored 0.

## A small model improves with training

We trained a 3-billion-parameter model (Qwen2.5-3B, LoRA) on the exact player's decisions and tested it on boards it never saw.

| Training steps | Score on the 30 task boards | Wins |
|---|---|---|
| 0 (untrained) | 0.000 | 0 |
| 2,400 | 0.417 | 6 |
| 6,883 | 0.811 | 24 |

So a small model trained on this game beats a frontier model that was not. A second model trained on seven rule sets kept its shot quality on rules it had never seen (0.93 against 0.45 untrained), including a 6x6 board.

The trade-off: this is a specialist. We did not measure its other abilities, and narrow training of this kind usually costs general ability. The frontier model is broad but plays from habit; the small model is precise here and nowhere else.

## How to run

```
uv sync
scripts/evaluate.sh                          # reference, do-nothing, stop-early on five boards
scripts/evaluate.sh -m PROVIDER/MODEL -k 3   # plus three agent runs per board
```

Run logs with scores are in `results/harbor-run-logs/`. Other scenarios we analysed (other boards, Claude models, series against an adapting opponent) are in `results/others/`.
