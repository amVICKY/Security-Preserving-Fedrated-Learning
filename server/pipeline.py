"""
Event-driven pipeline (pipes-and-filters) for an update travelling up the hierarchy.

The event is "a device finished a local epoch" — represented by UpdateEvent. It flows
through an ordered list of stages. Each stage inspects/annotates the event and returns
a StageResult: pass it on, or drop it (with a reason). The first drop short-circuits.

Stage vocabulary (Enterprise Integration Patterns):
  filter   -> drop too-stale / protocol-mismatch / duplicate
  splitter -> route to the right cluster
  merger   -> merge into FedAvg
  copier   -> duplicate to a backup leader
"""

from dataclasses import dataclass, field
from typing import Optional, Any

from communication.protocol import PROTOCOL_VERSION


@dataclass
class UpdateEvent:
    """The message that travels up the hierarchy."""
    raw: dict                                  # original payload
    node_id: Optional[str] = None
    cluster_id: Optional[str] = None
    lamport_ts: int = 0
    base_version: int = 0
    vector_clock: dict = field(default_factory=dict)
    update_id: Optional[str] = None
    weights: Any = None                        # deserialized delta

    # Filled in by stages as the event flows:
    staleness: int = 0
    local_ts: int = 0                          # coordinator's Lamport time after ClockStage
    dropped: bool = False
    status: Optional[str] = None               # final response status / drop reason


@dataclass
class StageResult:
    """A stage's verdict on an event."""
    passed: bool
    status: Optional[str] = None               # reason/status when not passed

    @classmethod
    def ok(cls):
        return cls(passed=True)

    @classmethod
    def drop(cls, status: str):
        return cls(passed=False, status=status)


class Pipeline:
    """Runs an event through an ordered list of stages, short-circuiting on the first drop."""

    def __init__(self, stages):
        self.stages = list(stages)

    def run(self, event: UpdateEvent, coord) -> UpdateEvent:
        for stage in self.stages:
            result = stage(event, coord)
            if not result.passed:
                event.dropped = True
                event.status = result.status
                return event
        return event


# ---------------------------------------------------------------------------
# Filter stages — each drops the event for one reason, or passes it on.
# ---------------------------------------------------------------------------

class ProtocolFilter:
    """Drop messages speaking a different protocol version."""
    def __call__(self, ev: UpdateEvent, coord) -> StageResult:
        if ev.raw.get("protocol_version") != PROTOCOL_VERSION:
            return StageResult.drop("protocol mismatch")
        return StageResult.ok()


class IdempotencyAppliedFilter:
    """Exactly-once (applied tier): drop a replay already merged into the model.
    model_vc[node] is the highest lamport_ts from this node already aggregated."""
    def __call__(self, ev: UpdateEvent, coord) -> StageResult:
        if ev.node_id and ev.lamport_ts <= coord.model_vc.get(ev.node_id, 0):
            return StageResult.drop("already applied")
        return StageResult.ok()


class ClockStage:
    """Transform (never drops): advance the coordinator's Lamport clock on receive."""
    def __call__(self, ev: UpdateEvent, coord) -> StageResult:
        if ev.lamport_ts:
            ev.local_ts = coord.lamport.update(ev.lamport_ts)
        else:
            ev.local_ts = coord.lamport.tick()
        return StageResult.ok()


class StalenessFilter:
    """Drop a delta computed against a base too far behind the current model."""
    def __call__(self, ev: UpdateEvent, coord) -> StageResult:
        ev.staleness = max(0, coord.model_version - ev.base_version)
        if ev.staleness > coord.max_staleness:
            coord._rejected_this_round += 1
            return StageResult.drop(
                f"rejected: staleness {ev.staleness} exceeds max {coord.max_staleness}"
            )
        return StageResult.ok()


class IdempotencyInflightFilter:
    """Exactly-once (in-flight tier): drop a duplicate already buffered this window."""
    def __call__(self, ev: UpdateEvent, coord) -> StageResult:
        if ev.update_id:
            if ev.update_id in coord._buffered_ids:
                return StageResult.drop("duplicate dropped")
            coord._buffered_ids.add(ev.update_id)
        return StageResult.ok()


# ---------------------------------------------------------------------------
# Splitter stage — route the event to the right cluster.
# ---------------------------------------------------------------------------

class ClusterRouter:
    """Defensive routing: an update tagged for a different cluster must not be merged
    into this coordinator's model. Unscoped cases (either side None) are accepted for
    backward compatibility with the single-cluster / global-server coordinators."""
    def __call__(self, ev: UpdateEvent, coord) -> StageResult:
        if coord.cluster_id is None or ev.cluster_id is None:
            return StageResult.ok()
        if ev.cluster_id != coord.cluster_id:
            return StageResult.drop(
                f"wrong cluster: update for {ev.cluster_id}, this leader serves {coord.cluster_id}"
            )
        return StageResult.ok()


# ---------------------------------------------------------------------------
# Copier stage — duplicate an accepted update to a backup leader (best-effort).
# ---------------------------------------------------------------------------

class BackupCopier:
    """Duplicate each accepted update to a backup sink so a leader crash mid-window
    doesn't lose in-flight updates. Placed LAST so only survivors (passed all filters)
    are copied. NEVER drops and NEVER raises — replication is a side-channel that must
    not affect the main aggregation path. The sink must itself be non-blocking."""
    def __init__(self, sink=None):
        self.sink = sink   # callable(raw_update_dict) -> None; non-blocking, best-effort

    def __call__(self, ev: UpdateEvent, coord) -> StageResult:
        if self.sink is not None:
            try:
                self.sink(ev.raw)
            except Exception:
                pass  # best-effort: a failed copy must never break the main path
        return StageResult.ok()
