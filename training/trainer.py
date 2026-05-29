import torch
from tqdm import tqdm

def train(model,train_loader,criterion,optimizer,device):

    model.train()
    running_loss = 0.0

    correct = 0
    total = 0

    progress_bar = tqdm(train_loader)

    for images,labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)

        correct += (predicted==labels).sum().item()
        progress_bar.set_description(
            f"Loss: {loss.item():.4f}"
        )
    
    epoch_loss = running_loss/len(train_loader)
    accuracy = 100 * correct / total
    print("inside trainer")
    return epoch_loss, accuracy
