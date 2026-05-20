from utils.config import load_config
from data.dataset import get_dataloaders

def main():

    config = load_config()

    print("Configuration Loaded Successfully")
    print(config)

    train_loader, test_loader = get_dataloaders(config)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

if __name__ == "__main__":
    main()

# print("Federated learning project initialized")