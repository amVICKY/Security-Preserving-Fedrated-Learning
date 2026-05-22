from fastapi import FastAPI
from server.model_manager import ModelManger
from communication.serialization import serialize_weights

app = FastAPI()
model_manager = ModelManger()

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
    
    print("\nReceived client update")
    return {
        "status":"update received"
    }

print("Hello")