import torch
import torch.nn as nn
import torch.optim as optim

from utils.config import load_config
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint
from utils.seed import set_seed

from data.dataset import get_dataloaders
from models.cnn import CNN

from training.trainer import train
from training.evaluator import evaluate


def main():

    print("="*50)
    config = load_config()
    logger = setup_logger()
    # device = config["training"]["device"]
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("="*50)

    set_seed(config["seed"])

    logger.info(f"Using Device: {device}")
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

    best_accuracy = 0.0

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

        logger.info(
            f"Epoch [{epoch+1}/epochs]"
            f"Train Loss: {train_loss:.4f}"
            f"Train Accuracy:{train_acc:.2f}%"
            f"Test Loss: {test_loss:.4f}"
            f"Test Accuracy: {test_acc:.2f}%"
        )

        if test_acc > best_accuracy:

            best_accuracy = test_acc
            save_checkpoint(
                model,
                "checkpoints/best_model.pth"
            )
            logger.info(
                f"Best model saved with accuracy: {best_accuracy:.2f}%"
            )

if __name__ == "__main__":
    main()