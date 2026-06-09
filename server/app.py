from fastapi import FastAPI
from server.coordinator import (
    FederatedCoordinator
)

app = FastAPI()
coordinator = FederatedCoordinator()
cluster_updates = []

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

@app.post("/aggregate")
def aggregate_updates():
    return coordinator.aggregate_updates()

@app.post("/cluster_update")
def cluster_update(update:dict):
    global cluster_updates
    cluster_updates.append(update)
    print(f"Received cluster update ({len(cluster_updates)})")
    return {
        "status":"received"
    }