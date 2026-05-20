import torch
from utils.config import load_config
from data.dataset import get_dataloaders
from models.cnn import CNN

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()

    train_loader, test_loader = get_dataloaders(config)

    model = CNN(
        num_classes=config["model"]["num_classes"]
    ).to(device)
    print(model)

    sample_batch = next(iter(train_loader))
    images,labels = sample_batch

    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)

    print(f"Device: {device}")
    print(f"Input Shape: {images.shape}")
    print(f"Ousstput Shape: {outputs.shape}")


if __name__ == "__main__":
    main()

# print("Federated learning project initialized")