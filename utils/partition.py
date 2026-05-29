from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np

from utils.config import load_config


configuration = load_config()

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

train_dataset = datasets.MNIST(
        root=configuration["paths"]["data_dir"],
        train=False,
        download=True,
        transform=transform
    )

def create_non_iid_partition(dataset, num_clients):
    
    labels = np.array(dataset.targets)
    num_target = len(np.unique(labels))
    client_indices = []
    classes_per_client = num_target // num_clients
    for client_id in range(num_clients):
        start_class = (client_id * classes_per_client) % num_target
        client_classes = list(
            range(
                start_class,
                start_class + classes_per_client
            )
        )
        client_classes = [
            c % num_target
            for c in client_classes
        ]
        indices = np.where(
            np.isin(labels, client_classes)
        )[0]
        client_indices.append(indices)
    return client_indices

    # print(classes_per_client)

create_non_iid_partition(train_dataset,9)