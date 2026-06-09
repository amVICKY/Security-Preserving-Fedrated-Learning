import sys
import requests
import torch
from models.cnn import CNN
from communication.serialization import (
    deserialize_weights,
    serialize_weights
)
from data.dataset import (
    get_dataloaders,
    get_client_dataloader
)
from utils.config import load_config
from utils.partition import create_non_iid_partition
from client.local_trainer import local_train
from communication.delta import compute_delta
from communication.model_sync import ModelSync


SERVER_URL = "http://127.0.0.1:8000"

class FederatedClient:

    def __init__(
        self,
        client_id,
        node,
        peer_table
    ):
        self.client_id = client_id
        self.node = node
        self.peer_table = peer_table
        self.config = load_config()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_url = self.resolve_target_url()

    def get_leader_url(self):
        leader = self.peer_table.get_cluster_leader(self.node.cluster_id)
        if leader is None:
            return None
        return (f"http://{leader.ip}:{leader.port}")

    def resolve_target_url(self):
        if self.node.role == "leader":
            return SERVER_URL
        if self.node.role == "worker":
            return self.get_leader_url()
        return None
    
    def download_global_model(self):
        target_url = self.resolve_target_url()
        if target_url is None:
            raise RuntimeError("Unable to locate target Node")
        response = requests.get(f"{target_url}/get_model")
        weights = response.json()["weights"]
        return deserialize_weights(weights)
    
    def upload_local_update(self,weights):
        target_url = self.resolve_target_url()
        if target_url is None:
            raise RuntimeError("Unable to loacate target node")
        serialize_weight = serialize_weights(weights)
        response = requests.post(
            f"{target_url}/send_update",
            json = {
                "weights":serialize_weight
            }
        )
        return response.json()

    def train_round(self):
        print("Downloading global model...")
        global_weights = (self.download_global_model())

        model = CNN(num_classes=10).to(self.device)
        model.load_state_dict(global_weights)
        initial_weights = {key:value.clone() for key,value in model.state_dict().items()}

        train_loader, test_loader = (get_dataloaders(self.config))
        train_dataset = train_loader.dataset
        num_clients = (self.config["federated"]["num_clients"])

        partitions = (create_non_iid_partition(train_dataset,num_clients))
        client_loader = (get_client_dataloader(self.config,train_dataset,partitions[self.client_id]))
        print("Starting local training")

        model,metrics = local_train(model,client_loader,test_loader,self.device,self.config["federated"]["local_epochs"])
        print("local training completed")
        print(metrics)

        delta = compute_delta(initial_weights,model.state_dict())
        print("Uploading local update")

        response = (self.upload_local_update(delta))
        print(response)

    def run(self):
        print(f"Node Role:{self.node.role}")
        print(f"Cluster:{self.node.cluster_id}")

        target_url = (self.resolve_target_url())
        print(f"Target URL:{target_url}")
        if target_url is None:
            print("No leader found")
            return 
        self.train_round()