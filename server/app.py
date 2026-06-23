import threading

import torch
from torch import nn
from fastapi import FastAPI
from server.coordinator import FederatedCoordinator
from server.global_aggregator import GlobalAggregator
from communication.serialization import deserialize_weights, serialize_weights
from communication.protocol import PROTOCOL_VERSION
from evalution.evaluate import evaluate_model
from data.dataset import get_dataloaders
from utils.config import load_config

app = FastAPI()
coordinator = FederatedCoordinator()           # holds the merged global model + does inference
global_aggregator = GlobalAggregator()          # merges per-cluster models (hierarchical FL)
registered_nodes = {}
_registry_lock = threading.Lock()  # guards registered_nodes against concurrent /register writes

# Durable backup sink (copier target): recent accepted updates per cluster, bounded.
backup_log = {}                    # cluster_id -> list[raw_update]
_backup_lock = threading.Lock()
BACKUP_MAX = 50

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
    node_id = node_info["node_id"]
    with _registry_lock:
        registered_nodes[node_id] = node_info

    print(f"Registered node: Node id:{node_id}| Role:{node_info.get('consensus_state')} | Cluster:{node_info.get('cluster_id')}")
    return {
        "status":"registered"
    }

@app.get("/registered_nodes")
def get_registered_nodes():
    with _registry_lock:
        return dict(registered_nodes)

@app.post("/backup_update")
def backup_update(update: dict):
    # Copier target: store a copy of an accepted update so a leader crash mid-window
    # doesn't lose in-flight updates. Bounded ring per cluster.
    cluster_id = update.get("cluster_id", "unknown")
    with _backup_lock:
        buf = backup_log.setdefault(cluster_id, [])
        buf.append(update)
        if len(buf) > BACKUP_MAX:
            del buf[:len(buf) - BACKUP_MAX]
        depth = len(buf)
    return {"status": "backed up", "cluster_id": cluster_id, "depth": depth}

@app.get("/backup_log")
def get_backup_log():
    # Shallow view (counts only, not the weights) for inspection.
    with _backup_lock:
        return {cid: len(buf) for cid, buf in backup_log.items()}


@app.post("/aggregate")
def aggregate_updates():
    return coordinator.aggregate_updates()

@app.post("/cluster_update")
def cluster_update(update: dict):
    if update.get("protocol_version") != PROTOCOL_VERSION:
        print(f"[APP] Protocol mismatch: received={update.get('protocol_version')} | expected={PROTOCOL_VERSION}")
        return {"status": "protocol mismatch"}

    cluster_id = update.get("cluster_id", "unknown")
    cluster_version = update.get("model_version", 0)
    weights = deserialize_weights(update["weights"])

    # Hierarchical merge: update this cluster's slot, re-average across all clusters
    global_weights, info = global_aggregator.receive_cluster_update(cluster_id, weights, cluster_version)
    coordinator.set_global_weights(global_weights)

    model = coordinator.model_manager.get_model().to(_device)
    metrics = evaluate_model(model, _test_loader, _criterion, _device)

    print(f"\n[GLOBAL AGG] cluster={cluster_id} pushed v{cluster_version} "
          f"| merging {info['num_clusters']} cluster(s) {info['cluster_versions']} "
          f"-> global_version {info['global_version']}")
    print(f"[GLOBAL INFERENCE] ========== Round Complete ==========")
    print(f"[GLOBAL INFERENCE] Test Accuracy : {metrics['accuracy']:.2f}%")
    print(f"[GLOBAL INFERENCE] Test Loss     : {metrics['loss']:.4f}")
    print(f"[GLOBAL INFERENCE] =====================================\n")

    return {
        "status": "inference complete",
        "global_version": info["global_version"],
        "num_clusters": info["num_clusters"],
        "test_accuracy": metrics["accuracy"],
        "test_loss": metrics["loss"],
        # Feedback loop: hand the merged global model back so the cluster can re-sync
        # to it and stay in the same basin as the other clusters (prevents drift).
        "global_weights": serialize_weights(global_weights)
    }