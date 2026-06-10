import torch
from torch import nn

from .model_manager import ModelManger
from communication.serialization import (
    serialize_weights,
    deserialize_weights
)

from server.aggregator import federated_averaging
from utils.config import load_config
from communication.delta import apply_delta
from evalution.evaluate import evaluate_model
from data.dataset import get_dataloaders

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
        self.train_loader,self.test_loader = (get_dataloaders(self.config))

        self.criterion = (nn.CrossEntropyLoss())
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

    def send_cluster_update(self,averaged_delta):
        try:
            reponse = ModelSync.upload_cluster_update(GLOBAL_SERVER_URL,averaged_delta)
        except Exception as e:
            print(f"Unable to send cluster update:{e}")

    def receive_update(self,update:dict):

        if(update["protocol_version"] != PROTOCOL_VERSION):
            return {
                "status":"protocol mismatch"
            }

        weights = deserialize_weights(update["weights"])
        self.client_updates.append(weights)
        print(f"\nReceived updates {len(self.client_updates)}/{self.num_clients}")
        
        if len(self.client_updates) >= self.num_clients:
            print("Aggregating global model")
            
            global_weights = (self.model_manager.get_weights())
            averaged_delta = (federated_averaging(self.client_updates))
            self.send_cluster_update(averaged_delta)
            updated_weights = (apply_delta(global_weights,averaged_delta))

            self.model_manager.set_weights(updated_weights)

            global_model = (self.model_manager.get_model().to(self.device))
            metrics = evaluate_model(
                self.model_manager.get_model(),
                self.test_loader,
                self.criterion,
                self.device
            )

            print(f"Global Loss:{metrics['loss']}")
            self.client_updates = []
            print("Global model Updated")
            
            return {
                "status":"aggregation completed"
            }
        
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