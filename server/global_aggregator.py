import threading

from server.aggregator import federated_averaging


class GlobalAggregator:
    """
    Top of the hierarchy: merges the per-cluster models into one global model.

    Each cluster leader pushes its FULL aggregated model (not a delta). We keep one
    slot per cluster and re-average all slots on every push — naturally asynchronous,
    no barrier, no deadlock. Because these are full weights (absolute positions, not
    displacements), cross-cluster FedAvg has none of the stale-delta hazard the
    intra-cluster layer must guard against.
    """

    def __init__(self):
        self.cluster_models: dict = {}    # cluster_id -> full model weights (state_dict)
        self.cluster_versions: dict = {}  # cluster_id -> that cluster's model_version
        self.global_version = 0           # bumped once per cross-cluster merge
        self._lock = threading.Lock()     # guards the dicts against concurrent cluster pushes

    def receive_cluster_update(self, cluster_id, weights, cluster_version):
        """Store this cluster's latest model and return the re-merged global model.

        Returns (global_weights, info) where info is metadata for logging.
        """
        with self._lock:
            self.cluster_models[cluster_id] = weights
            self.cluster_versions[cluster_id] = cluster_version

            models = list(self.cluster_models.values())
            global_weights = federated_averaging(models)

            self.global_version += 1
            info = {
                "global_version": self.global_version,
                "num_clusters": len(models),
                "cluster_versions": dict(self.cluster_versions),
                "triggered_by": cluster_id,
            }
            return global_weights, info
