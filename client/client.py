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

SERVER_URL = "http://127.0.0.1:8000"

def download_global_model():

    response = requests.get(
        f"{SERVER_URL}/get_model"
    )

    weights = response.json()["weights"]
    return deserialize_weights(weights)

def upload_local_update(weights):
    
    weights = serialize_weights(weights)
    respons = requests.post(
        f"{SERVER_URL}/send_update",
        json={"weights":weights}
    )
    return respons.json()

def main(client_id=0):
    
    config = load_config()
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Downloading global model..")
    global_weights = download_global_model()
    model = CNN(num_classes=10).to(device)
    model.load_state_dict(global_weights)

    initial_weights = {
        key:value.clone() for key, value in model.state_dict().items()
    }

    train_loader, test_loader = get_dataloaders(config)
    train_dataset = train_loader.dataset
    num_clients = config["federated"]["num_clients"]
    partitions = create_non_iid_partition(train_dataset,num_clients)
    client_loader = get_client_dataloader(config,train_dataset,partitions[client_id])

    print("Starting local training...")
    model, metrics = local_train(
        model,
        client_loader,
        test_loader,
        device,
        config["federated"]["local_epochs"]
    )

    print("Local training Completed")
    print(metrics)

    # response = upload_local_update(
    #     model.state_dict()
    # )

    print("Uploading local update(deltas)")
    delta = compute_delta(
        initial_weights,model.state_dict()
    )
    response = upload_local_update(delta)
    print(response)

if __name__ == "__main__":
    client_id = int(sys.argv[1])
    main(client_id)