import requests
from communication.serialization import (
    deserialize_weights,
    serialize_weights
)
from communication.protocol import  (
    PROTOCOL_VERSION
)

class ModelSync:
    
    @staticmethod
    def download_model(url):
        response = requests.get(f"{url}/get_model")
        payload = response.json()
        if payload["protocol_version"] != PROTOCOL_VERSION:
            raise RuntimeError("Protocol version mismatch")
        weights = payload["weights"]

        return deserialize_weights(weights)
    
    @staticmethod
    def upload_update(url, weights, node_id=None, lamport_ts=None, vector_clock=None):
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "weights": serialize_weights(weights),
        }
        if node_id is not None:
            payload["node_id"] = node_id
        if lamport_ts is not None:
            payload["lamport_ts"] = lamport_ts
        if vector_clock is not None:
            payload["vector_clock"] = vector_clock
        response = requests.post(f"{url}/send_update", json=payload)
        return response.json()
    
    @staticmethod
    def upload_cluster_update(url,delta):
        requests.post(
            f"{url}/cluster_update",
            json = {
                "protocol_version":PROTOCOL_VERSION,
                "weights":serialize_weights(delta)
            }
        )