"""
Prometheus metrics for the global server.

Grafana does not collect anything itself — it queries Prometheus, which scrapes
the `/metrics` endpoint exposed here. The global server is the natural single
source: it already knows the merged accuracy/loss, the global + per-cluster model
versions, and (via the registry) which nodes/leaders are alive.

Event metrics (accuracy, versions, update counters) are set from the request
handlers as things happen. State gauges (node/leader counts) are refreshed at
scrape time from the live registry via refresh_from_registry().
"""

from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST

# --- global model quality / progress ---
TEST_ACCURACY = Gauge("fl_test_accuracy_percent", "Global model test accuracy (%)")
TEST_LOSS = Gauge("fl_test_loss", "Global model test loss")
GLOBAL_VERSION = Gauge("fl_global_model_version", "Global (cross-cluster) model version")
CLUSTERS_ACTIVE = Gauge("fl_clusters_active", "Number of clusters merged into the global model")

# --- per-cluster ---
CLUSTER_VERSION = Gauge("fl_cluster_model_version", "Per-cluster model version", ["cluster_id"])
CLUSTER_UPDATES = Counter("fl_cluster_updates_total", "Cluster updates received by the global server", ["cluster_id"])

# --- topology (refreshed at scrape time from the registry) ---
NODES = Gauge("fl_registered_nodes", "Registered nodes currently known, by role", ["role"])


def refresh_from_registry(registered_nodes: dict):
    """Recompute node/leader counts from the live registry. Called on each scrape."""
    counts = {"leader": 0, "follower": 0, "unknown": 0}
    for info in registered_nodes.values():
        role = info.get("consensus_state") or "unknown"
        counts[role] = counts.get(role, 0) + 1
    for role, n in counts.items():
        NODES.labels(role=role).set(n)


def render():
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
