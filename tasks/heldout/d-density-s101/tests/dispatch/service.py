"""One episode behind the three public tools. Transport-independent.

Everything a controller can learn comes out of ``get_rules``, ``observe`` and
``step``. The action log written here is private: it is the only thing the
verifier trusts, and it scores the episode by replaying it.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .rules import rules
from .sim import CommandError, DispatchSim, RequestError, scenario_hash

MAX_REQUEST_IDS = 100_000


class EpisodeService:
    def __init__(self, scenario: dict, log_path: str | Path | None = None):
        self.sim = DispatchSim(scenario)
        self.lock = threading.Lock()
        self.answers: dict[str, dict] = {}
        self.log_path = Path(log_path) if log_path else None
        self.batches: list[list[dict]] = []
        self._score = 0.0
        self._write({"type": "header", "scenario": self.sim.id, "scenario_hash": scenario_hash(scenario)})

    def _write(self, rec: dict) -> None:
        if self.log_path is None:
            return
        rec["wall_time"] = round(time.time(), 3)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()

    # ---------------------------------------------------------------- tools
    def get_rules(self) -> dict:
        with self.lock:
            return rules(self.sim)

    def observe(self) -> dict:
        with self.lock:
            return self.sim.observation()

    def step(self, commands, expected_decision_id, request_id) -> dict:
        with self.lock:
            if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
                return self._refuse("invalid_request", "request_id must be a non-empty string (max 200 chars)")
            if request_id in self.answers:
                return {**self.answers[request_id], "duplicate_request": True}
            ans = self._step(commands, expected_decision_id)
            if ans["status"] == "accepted" and len(self.answers) < MAX_REQUEST_IDS:
                self.answers[request_id] = ans
            return ans

    def _refuse(self, status: str, message: str) -> dict:
        self._write({"type": "refused", "status": status, "message": message,
                     "decision_id": self.sim.decision_id})
        return {"status": status, "message": message, "time_advanced": False,
                "observation": self.sim.observation()}

    def _step(self, commands, expected_decision_id) -> dict:
        sim = self.sim
        if isinstance(commands, str):  # some clients send the list as a JSON string
            try:
                commands = json.loads(commands)
            except ValueError:
                return self._refuse("invalid_request", "commands is not valid JSON")
        if isinstance(expected_decision_id, bool) or not isinstance(expected_decision_id, int):
            return self._refuse("invalid_request", "expected_decision_id must be an integer")
        if sim.over:
            return self._refuse("episode_over", "the episode has ended; no further commands are accepted")
        if expected_decision_id != sim.decision_id:
            return self._refuse(
                "stale_decision",
                f"expected_decision_id {expected_decision_id} does not match the current decision_id "
                f"{sim.decision_id}; read the observation and resend")
        try:
            sim.apply(commands)  # validates the whole batch before changing anything
        except RequestError as e:
            return self._refuse("invalid_request", str(e))
        except CommandError as e:
            return self._refuse("rejected", str(e))
        self.batches.append(commands)
        sim.advance()
        score = sim.result()["score"]
        self._write({"type": "step", "decision_id": sim.decision_id, "commands": commands,
                     "tick_after": sim.time, "score_delta": round(score - self._score, 6),
                     "events": sim.events})
        self._score = score
        return {"status": "accepted", "message": "commands applied", "time_advanced": True,
                "observation": sim.observation()}


def load_batches(log_path: str | Path) -> tuple[dict | None, list[list[dict]]]:
    """Read a private action log. Returns (header, accepted command batches in order)."""
    header, batches = None, []
    for line in Path(log_path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") == "header" and header is None:
            header = rec
        elif rec.get("type") == "step":
            batches.append(rec["commands"])
    return header, batches
