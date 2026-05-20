import torch
import torch.nn as nn
import torch.optim as optim

from utils.config import load_config
from data.dataset import get_dataloaders
from models.cnn import CNN
from training.trainer import train
from training.evaluator import evaluate

def main():

    config = load_config()

    device = config["training"]["device"]

    train_loader, test_loader = get_dataloaders(config)

    model = CNN(
        num_classes=config["model"]["num_classes"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):

        train_loss, train_acc = train(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%\n")


if __name__ == "__main__":
    main()