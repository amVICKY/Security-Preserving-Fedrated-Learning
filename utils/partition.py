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

def create_non_iid_partition(dataset,num_clients):
    labels = np.array(dataset.targets)
    print(labels)
    print(len(labels))

create_non_iid_partition(train_dataset,9)