from fastapi import FastAPI
from server.coordinator import (
    FederatedCoordinator
)
from communication.protocol import (
    PROTOCOL_VERSION
)

app = FastAPI()
coordinator = FederatedCoordinator()
cluster_updates = []
registered_nodes = {}

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
def cluster_update(update:dict):

    version = update.get("protocol_version")
    if version != PROTOCOL_VERSION:
        print(f"Protocol mismatch: received={version} | expected={PROTOCOL_VERSION}")
        return {
            "status":"protocol mismatch"
        }

    global cluster_updates
    cluster_updates.append(update)
    print(f"Received cluster update ({len(cluster_updates)})")
    return {
        "status":"received"
    }

@app.get("/registered_updates")
def get_registered_updates():
    return cluster_updates