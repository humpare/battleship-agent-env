"""The rules text and static data every controller receives from get_rules()."""
from __future__ import annotations

RULES_VERSION = "dispatch-rules/1"

TEXT = """\
You are the dispatcher for a small railway. Bring every scheduled train to its
destination, as close to on time as you can, before the episode horizon.

NETWORK
- The network is a set of named sections joined at named junctions (see "network").
  Section kinds: "origin" (trains start here), "terminus" (trains finish here),
  "single" (one track shared by both directions), "loop" (one track of a passing loop).
- length_cells is the section length. A train occupies one cell and moves one cell per tick.
  A junction is one extra cell between sections.
- A junction "connects" pairs of sections. A train can pass only between a connected pair.

TRAINS
- Trains cannot reverse and cannot pass each other on the same track.
- Two trains never share a cell. A train whose next cell is occupied waits ("blocked").
- If two trains face each other on the same track, neither can ever move again (deadlock).
  Both are lost, and they block that track for everyone else until the episode ends.
- A train is off the network until it departs, and leaves the network when it arrives.
  It may not depart before its earliest_departure tick. Waiting off the network blocks nobody.
  A train with a route enters the network on the tick after its earliest_departure tick (or the
  tick after you give it a route, if later), on its origin section one cell in from the dead end.
  It has arrived when it reaches the cell one short of the dead end of its destination terminus.
- If two trains want the same free cell on the same tick, the one listed first in the timetable
  gets it. A train may move into a cell that another train leaves on the same tick.
- A malfunction stops a train where it is for a number of ticks. You see it when it starts,
  with malfunction_ticks_remaining = the number of further ticks it will not move.
  You get no warning in advance.

COMMANDS (per train; a train keeps its last order until you change it)
- {"train": ID, "action": "route", "route": [SECTION, ...]}
  The list must start with the section the train is in now (its origin if it has not departed)
  and continue through connected sections in its direction of travel. The train runs through
  them and stops at the far end of the last section, just before the next junction.
  A route ending in the train's destination terminus completes the journey.
  A new route replaces the old one. Trains may not enter an origin section or a terminus
  that is not their destination.
- {"action": "hold", "train": ID}
  The train stops where it is (or does not depart) until you give it a new route.
- Trains you do not mention keep following their current order.
- The simulator does NOT check that your routes are safe. Routes of different trains may
  conflict; the trains will then block each other or deadlock.

TIME
- Time only moves when you call step. Each step applies your commands, then runs until
  something happens (a train stops at the end of its route, is blocked, arrives, breaks down,
  recovers, or becomes ready to depart), at least 1 tick and at most {max_advance} ticks.
- The episode ends at tick {horizon}, or earlier when every train has arrived.
- If you stop calling step, the trains keep following their last orders until the horizon.

TOOLS
- get_rules(): this text, the network and the timetable.
- observe(): the current state. Does not advance time.
- step(commands, expected_decision_id, request_id): commands is a list (may be empty).
  expected_decision_id must equal the decision_id of the latest observation.
  request_id is any string unique to this call; repeating a request_id returns the first
  answer again without acting twice.
  A batch is applied only if every command in it is valid; otherwise nothing changes,
  time does not advance, and the answer says what was wrong.

SCORING
- Each train that arrives by its deadline counts 1.
- A train that arrives late loses credit in proportion to its lateness, down to 0.5 when it is
  {lateness_cap} or more ticks late (about one unobstructed end-to-end journey time).
- A train that has not arrived when the episode ends counts 0.
- Your result is the average over all trains. You are not shown your score.
"""


def rules(sim) -> dict:
    return {
        "version": RULES_VERSION,
        "rules": TEXT.replace("{max_advance}", str(sim.max_advance)).replace("{horizon}", str(sim.horizon))
                 .replace("{lateness_cap}", str(sim.lateness_cap)),
        "episode_id": sim.id,
        "horizon": sim.horizon,
        "max_advance_ticks": sim.max_advance,
        "lateness_cap_ticks": sim.lateness_cap,
        "network": sim.layout.public(),
        "timetable": [
            {"train": t.id, "origin": t.origin, "destination": t.destination,
             "earliest_departure": t.earliest_departure, "deadline": t.deadline}
            for t in sim.trains
        ],
    }
