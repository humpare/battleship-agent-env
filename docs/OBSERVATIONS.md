# Observations worth showing

Recorded 2026-11-14. Quotes are taken verbatim from the models' own end-of-game reports
(local arena, one-command client). Game records are under `results/`.

## 1. The models can run the algorithm by hand, and sometimes do

The exact algorithm lists every fleet still possible and shoots the cell that holds a ship in
the most of them. In endgames the models did the same thing, in their heads, with no tools.

**Opus 5.5, 5x5 single game 5 (the one it won):**
> "10 cells were left that could hold ships. I counted every legal way to fit two 2-cell ships
> and one 1-cell ship into them, all in my head. There were 49. D5 appeared in the most of them
> (35 of 49), so I fired there and it hit."
> "I recounted the 11 layouts left. A1 and A4 tied at 7 of 11, and I chose A1, which hit."

**Opus 5.5, 6x6 wrap-around, search only, game 1 (won):**
> "At first there were 291 layouts, and C1 and C5 were tied for best. I chose C1 and missed.
> After that miss, 173 layouts were left. C5 held a ship in 86 of them (about 50%), and it was a hit."

**Fable 5.2, 5x5 single game 4:**
> "I enumerated in my head every legal placement of the remaining ships and fired at the cell
> covered by the most of them (A5 at about 45/87, E2 at 17/34, C5 at 8/17, E1 at 6/9)."

**Fable 5.2, 5x5 single game 3** reasoned about the board before the first shot:
> "I shot corners first, on the informal judgement that the no-touching rule pushes fleets to the edges."

Why it matters: the capability is there. The opening failure is not an inability to compute; it
is using a remembered habit (hunting on a diagonal or checkerboard) when a derivation was needed.

## 2. At the opening they play from habit

- First shot C3: Opus 5/5 and Kimi 5/5 on 5x5; Opus 9/10 on the wrap-around board, where
  "centre" has no meaning. On the 5x5 board C3 itself is a reasonable shot (a ship in 52.3% of
  legal fleets, against 60.8% for a corner): it is predictable, not bad. The costly habit is
  what follows it.
- Opus then fired 8 of its 30 shots (5x5 single games) at B2/D2/B4/D4, cells that hold a ship in
  3.1% of legal fleets. The corners hold one in 60.8%.
- Opus on its own opening, 5x5 game 1: "I fired at C3 first because the center can hold the most
  placements of the 3-ship. Then I spread my shots diagonally (B2, D4)."

## 3. Given the history, Opus reads the opponent within one game

Series of five games, both sides placing, first version of the algorithm. Opus lost game 1 in
all three series and then won 10 of the next 12.

> "It always started with the four corners, then the middle of each edge, then the centre."
> "From game 2 on I kept my ships out of the corners, which cost it its first four shots every game."
> "It used the same fleet in games 1 and 2, and I wrote it down. In game 2 I aimed straight at
> those cells and sank everything with one miss."
> "I worked out that with this fleet on a 5x5 board you can't avoid every corner and every
> edge-middle cell. You have to give up either one corner or about three edge-middle cells."

## 4. Against the improved algorithm it sees pieces but not the whole

One series: Opus won 1 of 5 (1 of 4 after game 1).

> "It shoots cells I used in earlier games."
> "It places its fleet on edges, and in different spots each game... Its most-used cell was D1,
> but when I opened game 4 there, it had moved."

## 5. From the other side, exhaustive elimination looks like cheating

Several Opus players suspected the algorithm could see their board:

> "It fired 7 times in a row, and every one of those shots was on a ship... That looks far
> better than chance, which suggests the opponent may know where my ships are."
> "I can't tell whether this is clever modelling of me or partial knowledge of my board."

The audit of series-v2 game 4 reproduced all 14 of the algorithm's shots from public information
only. Its first four shots missed; after two hits only 4 fleets were still possible, then 3,
then 1, so its last six shots were certain.

One early case was a real artefact: before the tie-break was randomised, the algorithm broke
ties in board order starting at A1, and Opus anchored its fleets at A1. Results from that setup
(Opus, 6x6 wrap-around, own placement) are not used as evidence.

## 6. The models are honest about luck

> Fable: "Several shots were coin flips or modest-odds picks that went my way."
> Opus: "My mental counting was not exhaustive. I made some guesses on hunches... and some were wasted."

## 7. The models' hand counts are exactly right (checked against the algorithm)

Recomputed with the exact algorithm at the same positions (2026-11-14):

| Game, position | Model's count | Exact count | Model's shot | Exact best cell |
|---|---|---|---|---|
| Opus, 5x5 game 5, after 6 shots | 49 fleets, D5 in 35 | 49 fleets, D5 in 35 | D5 | D5 (71%) |
| Opus, 5x5 game 5, after 9 shots | 11 fleets, A1 and A4 in 7 each | 11 fleets, A1 and A4 in 7 each | A1 | A4 or A1 (64%) |
| Fable, 5x5 game 4, after 7 shots | 87 fleets, A5 in 45 | 87 fleets, A5 in 45 | A5 | A5 (52%) |
| Fable, 5x5 game 4, after 8 shots | 34 fleets, E2 in 17 | 34 fleets, E2 in 17 | E2 | E2 (50%) |
| Fable, 5x5 game 2, after 7 shots | 41 fleets, D1 in 25 | 41 fleets, D1 in 25 | D1 | E1 or D1 (61%) |
| Fable, 5x5 game 3, after 4 shots | 38 fleets, C3 most common | 38 fleets, C3 in 27 | C3 | C3 (71%) |

Six of six counts match to the unit, and every shot was a best cell. So once few enough fleets
remain (roughly under 100 here), the models perform the exact algorithm by hand, correctly.
At the opening there are 5,772 fleets, too many to list one by one; there the models fall back on
habit unless they reason structurally (as Fable did: "the no-touching rule pushes fleets to the edges").
