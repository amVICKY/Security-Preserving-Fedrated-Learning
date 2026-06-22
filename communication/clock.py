import threading


class LamportClock:
    """Scalar logical clock for total ordering and deduplication of updates."""

    def __init__(self):
        self._time = 0
        self._lock = threading.Lock()

    def tick(self) -> int:
        """Increment on send; returns the stamped time."""
        with self._lock:
            self._time += 1
            return self._time

    def update(self, received: int) -> int:
        """Advance on receive: max(local, received) + 1."""
        with self._lock:
            self._time = max(self._time, received) + 1
            return self._time

    @property
    def value(self) -> int:
        with self._lock:
            return self._time


class VectorClock:
    """Per-node vector clock for causality tracking and stale-update detection."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._clock: dict[str, int] = {}
        self._lock = threading.Lock()

    def tick(self) -> dict:
        """Increment own slot on send; returns a snapshot."""
        with self._lock:
            self._clock[self.node_id] = self._clock.get(self.node_id, 0) + 1
            return dict(self._clock)

    def update(self, received: dict):
        """Merge on receive: take element-wise max, then increment own slot."""
        with self._lock:
            all_nodes = set(self._clock) | set(received)
            for nid in all_nodes:
                self._clock[nid] = max(self._clock.get(nid, 0), received.get(nid, 0))
            self._clock[self.node_id] = self._clock.get(self.node_id, 0) + 1

    def happened_before(self, other: dict) -> bool:
        """True if self VC <= other VC and they differ (self causally precedes other)."""
        with self._lock:
            all_nodes = set(self._clock) | set(other)
            leq = all(self._clock.get(n, 0) <= other.get(n, 0) for n in all_nodes)
            equal = all(self._clock.get(n, 0) == other.get(n, 0) for n in all_nodes)
            return leq and not equal

    def is_concurrent(self, other: dict) -> bool:
        """True if neither VC dominates — both happened independently."""
        with self._lock:
            all_nodes = set(self._clock) | set(other)
            self_ahead = any(self._clock.get(n, 0) > other.get(n, 0) for n in all_nodes)
            other_ahead = any(other.get(n, 0) > self._clock.get(n, 0) for n in all_nodes)
            return self_ahead and other_ahead

    @property
    def value(self) -> dict:
        with self._lock:
            return dict(self._clock)
