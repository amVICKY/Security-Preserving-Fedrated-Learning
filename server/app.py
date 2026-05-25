import torch
from torch import nn
from fastapi import FastAPI
from server.model_manager import ModelManger
from communication.serialization import serialize_weights
from communication.serialization import deserialize_weights
from server.aggregator import federated_averaging
from utils.config import load_config
from communication.delta import apply_delta
from evalution.evaluate import evaluate_model
from data.dataset import get_dataloaders


app = FastAPI()
model_manager = ModelManger()
config = load_config()
client_updates = []

# ==========================================
device = ("cuda" if torch.cuda.is_available() else "cpu")
_, test_loader = get_dataloaders(config)
criterion = nn.CrossEntropyLoss()

NUM_CLIENTS = config["federated"]["num_clients"]

@app.get("/")
def root():
    return{
        "message":"Federated Learning Server Running"
    }

@app.get("/get_model")
def get_model():

    weights = model_manager.get_weights()
    weights = serialize_weights(weights)

    return {
        "weights":weights
    }

@app.post("/send_update")
def receive_update(update:dict):
    
    global client_updates
    weights = deserialize_weights(
        update["weights"]
    )

    client_updates.append(weights)
    print(
        f"\nReceived update"
        f"({len(client_updates)}/{NUM_CLIENTS})"
    )

    if len(client_updates) == NUM_CLIENTS:
        
        print("Aggregrating global model")
        global_weights = model_manager.get_weights()

        averaged_delta = federated_averaging(
            client_updates
        )
        updated_weights = apply_delta(
            global_weights,averaged_delta
        )
        model_manager.set_weights(
            updated_weights
        )

        global_model = model_manager.get_model().to(device)
        metrics = evaluate_model(
            model_manager.get_model(),
            test_loader,
            criterion,
            device
        )

        print(
            f"Global Accuracy: "
            f"{metrics['accuracy']:.2f}"
        )
        print(
            f"Global Loss: "
            f"{metrics['loss']:.4f}"
        )

        client_updates = [] # This reset the model weights
        print("Global model updated")
        
        return {
            "status":"aggregation completed"
        }
    
    return {
        "status":f"waiting for more clients, Client Updated:{len(client_updates)}, Client Remain: {NUM_CLIENTS-len(client_updates)} Client Total: {NUM_CLIENTS}"
    }

@app.post("/aggregate")
def aggregate_updates():
    global client_updates
    if len(client_updates) == 0:
        return{
            "status":"no updates received"
        }
    
    aggregated_weights = federated_averaging(
        client_updates
    )
    model_manager.set_weights(
        aggregated_weights
    )

    client_updates = []
    print("\nGlobal model updated")
    return {
        "status":"aggregation completed"
    }
print("Hello")