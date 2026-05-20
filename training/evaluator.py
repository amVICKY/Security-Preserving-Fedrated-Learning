import torch

def evaluate(model, test_loader, criterion, device):

    model.eval()
    running_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs,labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    
    epoch_loss = running_loss/ len(test_loader)
    accuracy = 100 * correct/total

    return epoch_loss, accuracy