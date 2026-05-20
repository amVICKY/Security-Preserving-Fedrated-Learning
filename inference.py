import torch
from torchvision import datasets, transforms

from models.cnn import CNN


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = CNN(num_classes=10).to(device)

    model.load_state_dict(
        torch.load(
            "checkpoints/best_model.pth",
            map_location=device
        )
    )

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    image, label = dataset[0]

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        prediction = torch.argmax(output, dim=1)

    print(f"Actual Label: {label}")
    print(f"Predicted Label: {prediction.item()}")


if __name__ == "__main__":
    main()