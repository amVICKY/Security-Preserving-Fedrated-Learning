import threading

import torch

from .model_manager import ModelManger
from communication.serialization import (
    serialize_weights,
    deserialize_weights
)

from server.aggregator import staleness_weighted_averaging
from utils.config import load_config
from communication.delta import apply_delta

from communication.model_sync import ModelSync
from communication.protocol import (
    PROTOCOL_VERSION
)
from communication.clock import LamportClock
from server.pipeline import (
    Pipeline,
    UpdateEvent,
    ProtocolFilter,
    ClusterRouter,
    IdempotencyAppliedFilter,
    ClockStage,
    StalenessFilter,
    IdempotencyInflightFilter,
    BackupCopier,
)

from utils.env import global_server_url

# Resolved from GLOBAL_SERVER_URL env var (defaults to localhost:8000 for loopback).
# leader_service.py imports this name, so keep it as a module-level constant.
GLOBAL_SERVER_URL = global_server_url()

class FederatedCoordinator:

    def __init__(self, cluster_id=None, num_workers=None):
        self.model_manager = ModelManger()
        self.client_updates = []  # list of (lamport_ts, base_version, staleness, node_id, weights)
        self.config = load_config()

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        # cluster identity (used when pushing to the global aggregator)
        self.cluster_id = cluster_id
        # Per-cluster worker count from the CLI; falls back to config if not given.
        # This is the "full batch" size that triggers immediate aggregation.
        self.num_clients = num_workers if num_workers is not None else self.config["federated"]["num_clients"]

        # Phase 4 — async aggregation parameters
        async_cfg = self.config["async_training"]
        self.min_updates = async_cfg["min_updates"]        # K: floor to arm the window timer
        self.window_seconds = async_cfg["window_seconds"]  # T: forced aggregation deadline
        # Staleness gate: drop updates whose base is more than this many versions behind.
        # A delta is only valid against the base it was computed from; applying a stale
        # delta to a moved-on model corrupts convergence. 0 = only base-consistent deltas.
        self.max_staleness = async_cfg["max_staleness"]

        self.model_version = 0           # bumped once per aggregation; sent to clients
        # Model vector clock: per-node highest lamport_ts that has been AGGREGATED into
        # the global model. Lets a client confirm (causally) that its own contribution
        # is baked in before it trains the next round — see client wait_for_merge().
        self.model_vc: dict = {}
        self.lamport = LamportClock()
        # In-flight idempotency: update_ids currently buffered (not yet aggregated).
        # The APPLIED-tier idempotency is model_vc itself (highest applied lamport_ts/node),
        # which is persistent — together they give exactly-once application.
        self._buffered_ids: set = set()
        self._rejected_this_round = 0    # stale updates dropped since last aggregation (metrics)

        self._lock = threading.Lock()    # guards client_updates, _buffered_ids, model_version, model
        self._timer = None               # active threading.Timer for the current window

        # Event-driven pipeline (pipes-and-filters): an incoming update flows through
        # protocol -> cluster routing (splitter) -> idempotency -> clock -> staleness
        # (filters), then survivors hit the merger (_merge_locked -> FedAvg).
        self.pipeline = Pipeline([
            ProtocolFilter(),
            ClusterRouter(),
            IdempotencyAppliedFilter(),
            ClockStage(),
            StalenessFilter(),
            IdempotencyInflightFilter(),
            BackupCopier(sink=self._backup_sink),   # last: only copies survivors
        ])

    def get_model(self):
        with self._lock:
            weights = self.model_manager.get_weights()
            version = self.model_version
            model_vc = dict(self.model_vc)
            weights = serialize_weights(weights)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "model_version": version,
            "model_vc": model_vc,
            "weights": weights
        }

    def get_status(self):
        """Lightweight model status (no weights) for clients polling causal merge state."""
        with self._lock:
            return {
                "protocol_version": PROTOCOL_VERSION,
                "model_version": self.model_version,
                "model_vc": dict(self.model_vc),
            }

    def set_global_weights(self, weights):
        with self._lock:
            self.model_manager.set_weights(weights)

    def receive_update(self, update: dict):
        # Build the event (the "device finished a local epoch" message) and run it
        # through the filter/splitter pipeline. Survivors reach the merger.
        node_id = update.get("node_id")
        lamport_ts = update.get("lamport_ts", 0)
        update_id = update.get("update_id") or (f"{node_id}:{lamport_ts}" if node_id else None)
        event = UpdateEvent(
            raw=update,
            node_id=node_id,
            cluster_id=update.get("cluster_id"),
            lamport_ts=lamport_ts,
            base_version=update.get("base_version", 0),
            vector_clock=update.get("vector_clock", {}),
            update_id=update_id,
        )

        updated_weights = None
        status = None

        with self._lock:
            self.pipeline.run(event, self)
            if event.dropped:
                short = node_id[:8] if node_id else "unknown"
                print(f"[COORDINATOR] Dropped update | reason='{event.status}' | node={short} | update_id={update_id}")
                return {"status": event.status, "update_id": update_id}

            updated_weights, status = self._merge_locked(event)

        # Network I/O happens outside the lock so it never blocks incoming updates
        if updated_weights is not None:
            self._push_global(updated_weights)

        return {"status": status}

    def _merge_locked(self, event: UpdateEvent):
        """Merger stage. MUST hold self._lock. Buffers the survivor update and triggers
        FedAvg when the full batch arrives. Returns (updated_weights_or_None, status)."""
        # Deserialize here (only for updates that survived the filters)
        weights = deserialize_weights(event.raw["weights"])
        self.client_updates.append(
            (event.lamport_ts, event.base_version, event.staleness, event.node_id, weights)
        )

        short_id = event.node_id[:8] if event.node_id else "unknown"
        print(
            f"\n[COORDINATOR] Update {len(self.client_updates)} buffered "
            f"| node={short_id} | lamport_ts={event.lamport_ts} | local_ts={event.local_ts} "
            f"| base_version={event.base_version} | current_version={self.model_version} "
            f"| staleness={event.staleness} | vc={event.vector_clock}"
        )

        if len(self.client_updates) >= self.num_clients:
            # Full batch — no reason to wait, aggregate immediately
            self._cancel_timer()
            return self._aggregate_locked(reason="full batch"), "aggregation completed"

        # Partial batch — arm the window timer so stragglers don't deadlock us
        if len(self.client_updates) >= self.min_updates:
            self._arm_timer()
        pending = len(self.client_updates)
        remaining = max(0, self.num_clients - pending)
        status = (
            f"Buffered | received={pending} | remaining={remaining} "
            f"| total={self.num_clients} | window={self.window_seconds}s"
        )
        return None, status

    def _on_window_timeout(self):
        """Timer callback: aggregate whatever arrived, dropping stragglers."""
        updated_weights = None
        with self._lock:
            self._timer = None
            if self.client_updates:
                updated_weights = self._aggregate_locked(
                    reason=f"{self.window_seconds}s window timeout"
                )
        if updated_weights is not None:
            self._push_global(updated_weights)

    def _aggregate_locked(self, reason=""):
        """Aggregate buffered updates. MUST be called with self._lock held.
        Returns the new global weights (for the caller to push outside the lock)."""
        # Deterministic order by Lamport timestamp
        self.client_updates.sort(key=lambda x: x[0])
        order = [u[0] for u in self.client_updates]
        staleness_list = [u[2] for u in self.client_updates]
        ordered_weights = [u[4] for u in self.client_updates]

        # Advance the model's vector clock: record the highest lamport_ts merged per node
        for lamport_ts, _base, _stale, node_id, _w in self.client_updates:
            if node_id:
                self.model_vc[node_id] = max(self.model_vc.get(node_id, 0), lamport_ts)

        avg_staleness = sum(staleness_list) / len(staleness_list)
        print(
            f"\n[COORDINATOR] Aggregating {len(ordered_weights)} updates ({reason}) "
            f"| lamport_order={order} | staleness={staleness_list} | avg_staleness={avg_staleness:.2f} "
            f"| stale_rejected={self._rejected_this_round}"
        )

        global_weights = self.model_manager.get_weights()
        averaged_delta, norm_weights = staleness_weighted_averaging(ordered_weights, staleness_list)
        updated_weights = apply_delta(global_weights, averaged_delta)
        self.model_manager.set_weights(updated_weights)

        self.model_version += 1
        rounded = [round(w, 3) for w in norm_weights]
        vc_short = {nid[:8]: ts for nid, ts in self.model_vc.items()}
        print(
            f"[COORDINATOR] Global model -> version {self.model_version} "
            f"| staleness_weights={rounded} | model_vc={vc_short}"
        )

        # Clear window state for the next batch. _buffered_ids resets (in-flight tier),
        # but model_vc persists as the applied-tier high-water-mark for exactly-once.
        self.client_updates = []
        self._buffered_ids.clear()
        self._rejected_this_round = 0
        return updated_weights

    def _arm_timer(self):
        """Start the forced-aggregation timer if one isn't already running."""
        if self._timer is None:
            self._timer = threading.Timer(self.window_seconds, self._on_window_timeout)
            self._timer.daemon = True
            self._timer.start()
            print(f"[COORDINATOR] Aggregation window opened — {self.window_seconds}s until forced aggregation")

    def _cancel_timer(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _backup_sink(self, raw_update):
        """Non-blocking, best-effort copy of an accepted update to the durable backup
        sink. Fire-and-forget on a daemon thread so the main path is never blocked."""
        def _send():
            try:
                ModelSync.upload_backup(GLOBAL_SERVER_URL, raw_update, cluster_id=self.cluster_id)
            except Exception:
                pass  # best-effort
        threading.Thread(target=_send, daemon=True).start()

    def _push_global(self, updated_weights):
        print(f"[COORDINATOR] Global model updated — pushing cluster={self.cluster_id} v{self.model_version} to global aggregator")
        try:
            result = ModelSync.upload_cluster_update(
                GLOBAL_SERVER_URL,
                updated_weights,
                cluster_id=self.cluster_id,
                model_version=self.model_version
            )
            # Feedback loop: adopt the merged global model so this cluster re-syncs
            # toward the other clusters and stays in the shared weight-space basin.
            global_weights = result.get("global_weights") if isinstance(result, dict) else None
            if global_weights is not None:
                self.set_global_weights(global_weights)
                gv = result.get("global_version")
                print(f"[COORDINATOR] Re-synced to global model (global_version={gv}, clusters={result.get('num_clusters')})")
        except Exception as e:
            print(f"[COORDINATOR] app.py unreachable, skipping global sync: {e}")

    def aggregate_updates(self):
        """Manual aggregation trigger (POST /aggregate). Aggregates whatever is buffered."""
        updated_weights = None
        with self._lock:
            if not self.client_updates:
                return {"status": "no updates received"}
            self._cancel_timer()
            updated_weights = self._aggregate_locked(reason="manual trigger")
        if updated_weights is not None:
            self._push_global(updated_weights)
        return {"status": "aggregation completed"}


if __name__ == "__main__":
    coordinator = FederatedCoordinator()
    print(coordinator.get_model())
    print("Smoke test done")
