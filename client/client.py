import requests
import torch
from models.cnn import CNN
from communication.serialization import (
    deserialize_weights,
    serialize_weights
)
from data.dataset import get_dataloaders
from utils.config import load_config
from client.local_trainer import local_train

# SERVER_URL = "http://127.0.0.1:8000"
from dotenv import load_dotenv


import os

load_dotenv()

SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = os.getenv("SERVER_PORT", "8000")

SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
import time
import requests

def download_global_model():

    max_retries = 10

    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"{SERVER_URL}/get_model",
                timeout=10
            )

            weights = response.json()["weights"]
            return deserialize_weights(weights)

        except requests.exceptions.RequestException:

            print(
                f"Server not ready "
                f"({attempt+1}/{max_retries})"
            )

            time.sleep(3)

    raise RuntimeError(
        "Unable to connect to server"
    )

def upload_local_update(weights):
    
    weights = serialize_weights(weights)
    respons = requests.post(
        f"{SERVER_URL}/send_update",
        json={"weights":weights}
    )
    return respons.json()

def main():

    client_id = os.getenv("HOSTNAME", "unknown")
    print(f"\nClient ID: {client_id}")
    config = load_config()
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Downloading global model..")
    global_weights = download_global_model()
    model = CNN(num_classes=10).to(device)
    model.load_state_dict(global_weights)
    train_loader, test_loader = get_dataloaders(config)
    print("Starting local training...")
    model, metrics = local_train(
        model,
        train_loader,
        test_loader,
        device,
        config["federated"]["local_epochs"]
    )

    print("Local training Completed")
    print(metrics)

    print("Uploading local update")
    response = upload_local_update(
        model.state_dict()
    )

    print(response)

if __name__ == "__main__":
    main()