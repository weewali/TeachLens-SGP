"""Join tracker IDs of one person walking between one-per-second samples.

ByteTrack links a box to the previous frame's boxes by overlap. At one
analyzed frame per second a walking person moves more than a box width, the
boxes do not overlap, and ByteTrack starts a new ID: a student walking out of
the room, to a classmate, or back to the seat became two or three "people",
each with its own apparent entry and exit (A-scenario note, 2026-11-17).
Looser ByteTrack matching (configs/bytetrack_1fps.yaml) helps for small
moves but cannot link boxes that do not touch.

TrackStitcher runs after ByteTrack: when a new tracker ID appears, it looks
for a person lost within max_gap_seconds who could have walked there (at most
max_speed box heights per second from where they were heading, plus a
margin) and has a similar box height. If exactly one such person fits
clearly better than any other, the new ID takes the old ID. Ambiguous cases
(two lost people near the new box) are left alone: a wrong join would merge
two people, which is worse than a split.
"""

from dataclasses import dataclass, replace
import math
from typing import Sequence

from src.inference.pose import PosePerson


@dataclass
class _Last:
    time: float
    center: tuple[float, float]   # pixels
    height: float                 # box height, pixels
    velocity: tuple[float, float] | None = None  # pixels per second


class TrackStitcher:
    """Map raw tracker IDs to stable IDs; call once per analyzed sample."""

    def __init__(self, max_gap_seconds: float = 2.5, max_speed: float = 1.5,
                 margin: float = 0.5, min_height_ratio: float = 0.6, ambiguity_ratio: float = 1.5):
        self.max_gap = max_gap_seconds
        self.max_speed = max_speed          # box heights per second
        self.margin = margin                # box heights allowed on top of the walk
        self.min_ratio = min_height_ratio   # smaller / larger box height
        self.ambiguity = ambiguity_ratio    # runner-up must score this much worse
        self.alias: dict[int, int] = {}     # raw ID -> stable ID
        self.last: dict[int, _Last] = {}    # stable ID -> last observation
        self.stitched = 0                   # joins made, for the run log

    @staticmethod
    def _observe(person: PosePerson) -> tuple[tuple[float, float], float]:
        x1, y1, x2, y2 = person.box
        return ((x1 + x2) / 2 * person.width, (y1 + y2) / 2 * person.height), (y2 - y1) * person.height

    def _score(self, last: _Last, center: tuple[float, float], height: float,
               time: float) -> float | None:
        """Distance in units of the allowed distance (<= 1 fits), or None."""
        dt = time - last.time
        if not 0 < dt <= self.max_gap + 1e-6:
            return None
        if min(height, last.height) / max(height, last.height) < self.min_ratio:
            return None
        scale = max(height, last.height)
        vx, vy = last.velocity or (0.0, 0.0)
        cap = self.max_speed * scale  # trust the heading only up to walking speed
        speed = math.hypot(vx, vy)
        if speed > cap:
            vx, vy = vx * cap / speed, vy * cap / speed
        predicted = (last.center[0] + vx * dt, last.center[1] + vy * dt)
        allowed = (self.max_speed * dt + self.margin) * scale
        score = math.dist(center, predicted) / allowed
        return score if score <= 1 else None

    def __call__(self, time: float, people: Sequence[PosePerson]) -> tuple[PosePerson, ...]:
        assigned: dict[int, int] = {}  # index in people -> stable ID
        used: set[int] = set()
        fresh = []
        for index, person in enumerate(people):
            stable = self.alias.get(person.track_id, person.track_id if person.track_id in self.last else None)
            if stable is not None and stable not in used:
                assigned[index] = stable
                used.add(stable)
            else:
                fresh.append(index)
        lost = {sid: last for sid, last in self.last.items() if sid not in used}
        for index in fresh:
            person = people[index]
            center, height = self._observe(person)
            scored = sorted((score, sid) for sid, last in lost.items()
                            if (score := self._score(last, center, height, time)) is not None)
            stable = person.track_id
            if scored and (len(scored) == 1 or scored[1][0] >= self.ambiguity * max(scored[0][0], 1e-6)):
                stable = scored[0][1]
                self.stitched += 1
                del lost[stable]
            if stable in used:  # e.g. ByteTrack revived an ID already joined; keep them apart
                stable = person.track_id
            self.alias[person.track_id] = stable
            assigned[index] = stable
            used.add(stable)
        output = []
        for index, person in enumerate(people):
            stable = assigned[index]
            center, height = self._observe(person)
            previous = self.last.get(stable)
            velocity = None
            if previous is not None and 0 < time - previous.time <= self.max_gap:
                dt = time - previous.time
                velocity = ((center[0] - previous.center[0]) / dt, (center[1] - previous.center[1]) / dt)
            self.last[stable] = _Last(time, center, height, velocity)
            output.append(person if stable == person.track_id else replace(person, track_id=stable))
        # Forget people lost for longer than the gap: they cannot be joined any more.
        for sid in [sid for sid, last in self.last.items() if time - last.time > self.max_gap]:
            del self.last[sid]
        return tuple(output)
