# Railway dispatching

You are the dispatcher for a small railway. A simulator is connected to you as the MCP
server `dispatch`. It is the only thing you can act on.

Your job: bring every scheduled train to its destination, as close to on time as you can,
before the episode ends.

How to work:

1. Call `get_rules` once. It gives you the full rules, the command format, the track network
   and the timetable. Read it carefully: the simulator does not stop you from sending trains
   into each other.
2. Call `observe` to see the current state.
3. Call `step` with your commands. Time only moves when you call `step`. Each call returns the
   new state. Keep going until the observation says `episode_over: true`.

There is one episode and no reset. Whatever is unfinished when you stop, or when the episode
ends, counts against you. You have no other tools and need none.
