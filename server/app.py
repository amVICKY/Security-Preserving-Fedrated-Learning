import torch
from torch import nn
from fastapi import FastAPI
from server.coordinator import FederatedCoordinator
from communication.serialization import deserialize_weights
from communication.protocol import PROTOCOL_VERSION
from evalution.evaluate import evaluate_model
from data.dataset import get_dataloaders
from utils.config import load_config

app = FastAPI()
coordinator = FederatedCoordinator()
registered_nodes = {}

_config = load_config()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_, _test_loader = get_dataloaders(_config)
_criterion = nn.CrossEntropyLoss()

@app.get("/")
def root():
    return{
        "message":"Federated Learning Server Running"
    }

@app.get("/get_model")
def get_model():
    return coordinator.get_model()

@app.post("/send_update")
def receive_update(update:dict):
    return coordinator.receive_update(update)

@app.post("/register")
def register_node(node_info:dict):
    global registered_nodes
    node_id = node_info["node_id"]
    registered_nodes[node_id] = node_info

    print(f"Registered node: Node id:{node_id}| Role:{node_info.get('consensus_state')} | Cluster:{node_info.get('cluster_id')}")
    return {
        "status":"registered"
    }

@app.get("/registered_nodes")
def get_registered_nodes():
    return registered_nodes


@app.post("/aggregate")
def aggregate_updates():
    return coordinator.aggregate_updates()

@app.post("/cluster_update")
def cluster_update(update: dict):
    if update.get("protocol_version") != PROTOCOL_VERSION:
        print(f"[APP] Protocol mismatch: received={update.get('protocol_version')} | expected={PROTOCOL_VERSION}")
        return {"status": "protocol mismatch"}

    weights = deserialize_weights(update["weights"])
    coordinator.set_global_weights(weights)

    model = coordinator.model_manager.get_model().to(_device)
    metrics = evaluate_model(model, _test_loader, _criterion, _device)

    print(f"\n[GLOBAL INFERENCE] ========== Round Complete ==========")
    print(f"[GLOBAL INFERENCE] Test Accuracy : {metrics['accuracy']:.2f}%")
    print(f"[GLOBAL INFERENCE] Test Loss     : {metrics['loss']:.4f}")
    print(f"[GLOBAL INFERENCE] =====================================\n")

    return {
        "status": "inference complete",
        "test_accuracy": metrics["accuracy"],
        "test_loss": metrics["loss"]
    }