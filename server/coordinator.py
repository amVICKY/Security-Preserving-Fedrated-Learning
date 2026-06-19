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

GLOBAL_SERVER_URL = "http://127.0.0.1:8000"

class FederatedCoordinator:
    
    def __init__(self):
        self.model_manager = ModelManger()
        self.client_updates = []
        self.config = load_config()
        
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_clients = (self.config["federated"]["num_clients"])

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

    def receive_update(self,update:dict):

        if(update["protocol_version"] != PROTOCOL_VERSION):
            return {
                "status":"protocol mismatch"
            }

        weights = deserialize_weights(update["weights"])
        self.client_updates.append(weights)
        print(f"\nReceived updates {len(self.client_updates)}/{self.num_clients}")
        
        if len(self.client_updates) >= self.num_clients:
            print(f"\n[COORDINATOR] All {self.num_clients} updates received — aggregating")

            global_weights = self.model_manager.get_weights()
            averaged_delta = federated_averaging(self.client_updates)
            updated_weights = apply_delta(global_weights, averaged_delta)
            self.model_manager.set_weights(updated_weights)

            self.client_updates = []
            print(f"[COORDINATOR] Global model updated — sending to app.py for inference")

            try:
                ModelSync.upload_cluster_update(GLOBAL_SERVER_URL, updated_weights)
            except Exception as e:
                print(f"[COORDINATOR] app.py unreachable, skipping global inference: {e}")

            return {"status": "aggregation completed"}
        
        return {
            "status":f"Waiting for more clients| Client Updated:{len(self.client_updates)}| Client Remain:{self.num_clients-len(self.client_updates)}| Client Total:{self.num_clients}"
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