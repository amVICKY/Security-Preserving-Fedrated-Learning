from fastapi import FastAPI
from server.model_manager import ModelManger
from communication.serialization import serialize_weights
from communication.serialization import deserialize_weights
from server.aggregator import federated_averaging

app = FastAPI()
model_manager = ModelManger()
client_updates = []

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
        f"from client"
        f"({len(client_updates)} total)"
    )

    return {
        "status":"update received"
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