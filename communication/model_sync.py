from communication.serialization import (
    deserialize_weights,
    serialize_weights
)
from communication.protocol import  (
    PROTOCOL_VERSION
)
from communication.resiliency import resilient_request

class ModelSync:

    @staticmethod
    def download_model(url):
        response = resilient_request("GET", f"{url}/get_model")
        payload = response.json()
        if payload["protocol_version"] != PROTOCOL_VERSION:
            raise RuntimeError("Protocol version mismatch")
        weights = deserialize_weights(payload["weights"])
        # Defaults to 0 until the coordinator starts emitting model_version (Step 4)
        model_version = payload.get("model_version", 0)

        return weights, model_version

    @staticmethod
    def upload_backup(url, raw_update, cluster_id=None):
        """Best-effort copy of an accepted update to a durable backup sink."""
        payload = dict(raw_update)
        if cluster_id is not None:
            payload.setdefault("cluster_id", cluster_id)
        # Best-effort: short timeout, single attempt — must never delay the main path.
        resilient_request("POST", f"{url}/backup_update", json=payload, timeout=2, max_retries=1)

    @staticmethod
    def get_model_status(url):
        """Cheap poll of (model_version, model_vc) without downloading weights."""
        # Polled in a loop (wait_for_merge), so keep retries low.
        response = resilient_request("GET", f"{url}/model_status", max_retries=1)
        payload = response.json()
        if payload["protocol_version"] != PROTOCOL_VERSION:
            raise RuntimeError("Protocol version mismatch")
        return payload["model_version"], payload.get("model_vc", {})

    @staticmethod
    def upload_update(url, weights, node_id=None, lamport_ts=None, vector_clock=None, base_version=None, update_id=None, cluster_id=None):
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "weights": serialize_weights(weights),
        }
        if node_id is not None:
            payload["node_id"] = node_id
        if cluster_id is not None:
            payload["cluster_id"] = cluster_id
        if lamport_ts is not None:
            payload["lamport_ts"] = lamport_ts
        if vector_clock is not None:
            payload["vector_clock"] = vector_clock
        if base_version is not None:
            payload["base_version"] = base_version
        if update_id is not None:
            payload["update_id"] = update_id
        # Safe to retry: deduped by update_id at the aggregator (exactly-once).
        response = resilient_request("POST", f"{url}/send_update", json=payload)
        return response.json()
    
    @staticmethod
    def upload_cluster_update(url, delta, cluster_id=None, model_version=None):
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "weights": serialize_weights(delta),
        }
        if cluster_id is not None:
            payload["cluster_id"] = cluster_id
        if model_version is not None:
            payload["model_version"] = model_version
        # Safe to retry: per-cluster slot is last-write-wins (idempotent).
        response = resilient_request("POST", f"{url}/cluster_update", json=payload)
        result = response.json()
        # Feedback loop: parse the merged global model the server hands back (if any).
        if "global_weights" in result:
            result["global_weights"] = deserialize_weights(result["global_weights"])
        return result