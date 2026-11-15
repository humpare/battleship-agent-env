# Battleship

You are playing Battleship on a 5x5 board against a coded opponent. A game server is
connected to you as the MCP server `battleship`. It is the only thing you can act on.

Your job: sink the opponent's whole fleet before it sinks yours.

1. Call `get_rules` once and read it carefully: the rules differ from the usual game (ships may
   not touch), and they tell you exactly how the opponent places its fleet and chooses its shots.
2. Call `place_fleet` to place your own five ships.
3. Call `fire` with one target cell at a time. A hit lets you shoot again; a miss gives the
   opponent its turn, and you see its shots in the answer. Keep going until the observation
   says `game_over: true`.

There is one game and no reset. You have no other tools and need none.
