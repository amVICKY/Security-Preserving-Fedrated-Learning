import torch

from .model_manager import ModelManger
from communication.serialization import (
    serialize_weights,
    deserialize_weights
)

from server.aggregator import federated_averaging
from utils.config import load_config
from communication.delta import apply_delta

from communication.model_sync import ModelSync
from communication.protocol import (
    PROTOCOL_VERSION
)
from communication.clock import LamportClock

GLOBAL_SERVER_URL = "http://127.0.0.1:8000"

class FederatedCoordinator:
    
    def __init__(self):
        self.model_manager = ModelManger()
        self.client_updates = []  # list of (lamport_ts, weights)
        self.config = load_config()

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_clients = (self.config["federated"]["num_clients"])
        self.lamport = LamportClock()
        self._seen_updates: set = set()  # (node_id, lamport_ts) pairs seen this round

    def get_model(self):
        weights = (
            self.model_manager.get_weights()
        )
        weights = serialize_weights(weights)
        return {
            "protocol_version":PROTOCOL_VERSION,
            "weights":weights
        }
    
    def set_global_weights(self,weights):
        self.model_manager.set_weights(weights)

    def receive_update(self, update: dict):

        if update["protocol_version"] != PROTOCOL_VERSION:
            return {"status": "protocol mismatch"}

        node_id = update.get("node_id")
        lamport_ts = update.get("lamport_ts", 0)
        vector_clock = update.get("vector_clock", {})

        # Advance coordinator's Lamport clock
        local_ts = self.lamport.update(lamport_ts) if lamport_ts else self.lamport.tick()

        # Dedup: drop retried or replayed updates within this round
        if node_id:
            dedup_key = (node_id, lamport_ts)
            if dedup_key in self._seen_updates:
                print(f"[COORDINATOR] Duplicate dropped | node={node_id[:8]} | lamport_ts={lamport_ts}")
                return {"status": "duplicate dropped"}
            self._seen_updates.add(dedup_key)

        weights = deserialize_weights(update["weights"])
        self.client_updates.append((lamport_ts, weights))

        short_id = node_id[:8] if node_id else "unknown"
        print(
            f"\n[COORDINATOR] Update {len(self.client_updates)}/{self.num_clients} "
            f"| node={short_id} | lamport_ts={lamport_ts} | local_ts={local_ts} | vc={vector_clock}"
        )

        if len(self.client_updates) >= self.num_clients:
            # Sort by Lamport timestamp so aggregation order is deterministic
            self.client_updates.sort(key=lambda x: x[0])
            order = [ts for ts, _ in self.client_updates]
            print(f"\n[COORDINATOR] All {self.num_clients} updates received — aggregating in Lamport order {order}")

            ordered_weights = [w for _, w in self.client_updates]
            global_weights = self.model_manager.get_weights()
            averaged_delta = federated_averaging(ordered_weights)
            updated_weights = apply_delta(global_weights, averaged_delta)
            self.model_manager.set_weights(updated_weights)

            self.client_updates = []
            self._seen_updates.clear()
            print(f"[COORDINATOR] Global model updated — sending to app.py for inference")

            try:
                ModelSync.upload_cluster_update(GLOBAL_SERVER_URL, updated_weights)
            except Exception as e:
                print(f"[COORDINATOR] app.py unreachable, skipping global inference: {e}")

            return {"status": "aggregation completed"}

        remaining = self.num_clients - len(self.client_updates)
        return {
            "status": f"Waiting for more clients | received={len(self.client_updates)} | remaining={remaining} | total={self.num_clients}"
        }

    def aggregate_updates(self):
        if len(self.client_updates)==0:
            return {
                "status":"no updates received"
            }
        aggregated_weights = (federated_averaging(self.client_updates))
        self.model_manager.set_weights(aggregated_weights)
        self.client_updates = []
        print("Global model updated")

        return {
            "status":"aggregation completed"
        }
        

if __name__ == "__main__":
    coordinator = FederatedCoordinator()
    print(coordinator.get_model())
    print("Smoke test done")