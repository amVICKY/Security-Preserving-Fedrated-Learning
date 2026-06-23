"""
Offline smoke test — verifies the Phase 4/5 logic in-process (no cluster needed).

Run:  python smoke_test.py
Exit code 0 = all green, 1 = something failed.

Covers: logical clocks, the event pipeline (filters/splitter/copier/merger),
async aggregation, versioning, staleness gate, exactly-once dedup, and the
cross-cluster global aggregator. The live multi-cluster run covers the rest
(Raft, discovery, HTTP, convergence).
"""

import io
import sys
import contextlib

import torch

from communication.protocol import PROTOCOL_VERSION
from communication.serialization import serialize_weights
from communication.clock import LamportClock, VectorClock
from server.coordinator import FederatedCoordinator
from server.global_aggregator import GlobalAggregator

_results = []


def check(name, cond):
    _results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


@contextlib.contextmanager
def quiet():
    """Silence the coordinator's verbose prints during assertions."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def upd(node, ts, base, cid="cluster1", proto=PROTOCOL_VERSION, weights_ser=None):
    return {
        "protocol_version": proto,
        "weights": weights_ser,
        "node_id": node,
        "lamport_ts": ts,
        "vector_clock": {node: ts},
        "base_version": base,
        "update_id": f"{node}:{ts}",
        "cluster_id": cid,
    }


print("== logical clocks ==")
lc = LamportClock()
lc.tick()
check("lamport tick -> 1", lc.value == 1)
check("lamport update max+1", lc.update(5) == 6)
vc = VectorClock("a")
vc.tick()
check("vector clock tick", vc.value == {"a": 1})

print("\n== coordinator event pipeline ==")
c = FederatedCoordinator(cluster_id="cluster1", num_workers=2)
ser = serialize_weights(c.model_manager.get_weights())

with quiet():
    c.receive_update(upd("alice", 1, 0, weights_ser=ser))
    c.receive_update(upd("bob", 1, 0, weights_ser=ser))
check("two-worker full-batch aggregation -> v1", c.model_version == 1)
check("model_vc records both contributions", c.model_vc.get("alice") == 1 and c.model_vc.get("bob") == 1)

with quiet():
    r = c.receive_update(upd("alice", 1, 0, weights_ser=ser))
check("exactly-once: applied-tier replay dropped", r["status"] == "already applied")

with quiet():
    r = c.receive_update(upd("alice", 2, 0, weights_ser=ser))
check("staleness gate drops base-0 vs v1", "rejected" in r["status"])

with quiet():
    r = c.receive_update(upd("carol", 5, 1, cid="cluster2", weights_ser=ser))
check("splitter drops wrong-cluster update", "wrong cluster" in r["status"])

with quiet():
    r = c.receive_update(upd("alice", 9, 1, proto=999, weights_ser=ser))
check("protocol filter drops mismatch", r["status"] == "protocol mismatch")

with quiet():
    r1 = c.receive_update(upd("bob", 2, 1, weights_ser=ser))   # buffers, arms timer
    r2 = c.receive_update(upd("bob", 2, 1, weights_ser=ser))   # in-flight dup
check("exactly-once: in-flight duplicate dropped", r2["status"] == "duplicate dropped")
check("buffer holds 1 (not double-counted)", len(c.client_updates) == 1)
c._cancel_timer()

print("\n== copier (backup) ==")
c2 = FederatedCoordinator(cluster_id="cluster1", num_workers=2)
copied = []
c2.pipeline.stages[-1].sink = lambda raw: copied.append(raw.get("update_id"))
with quiet():
    c2.receive_update(upd("alice", 1, 0, weights_ser=ser))
    c2.receive_update(upd("bob", 1, 0, weights_ser=ser))
check("copier duplicated both survivors", copied == ["alice:1", "bob:1"])
with quiet():
    c2.receive_update(upd("alice", 1, 0, weights_ser=ser))   # replay -> dropped pre-copier
check("copier skips dropped (replay) updates", copied == ["alice:1", "bob:1"])
c2._cancel_timer()

print("\n== global aggregator (cross-cluster) ==")
ga = GlobalAggregator()
A = {"w": torch.tensor([0.0, 10.0])}
B = {"w": torch.tensor([20.0, 0.0])}
g, _ = ga.receive_cluster_update("cluster1", A, 1)
check("1 cluster -> passthrough (no regression)", g["w"].tolist() == [0.0, 10.0])
g, info = ga.receive_cluster_update("cluster2", B, 1)
check("2 clusters -> FedAvg merge", g["w"].tolist() == [10.0, 5.0])
check("tracks per-cluster versions", info["cluster_versions"] == {"cluster1": 1, "cluster2": 1})

passed = sum(_results)
total = len(_results)
print(f"\n{'='*40}\n{passed}/{total} checks passed")
sys.exit(0 if passed == total else 1)
