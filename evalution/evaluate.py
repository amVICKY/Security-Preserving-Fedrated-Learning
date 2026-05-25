import torch

def evaluate_model(model,test_loader,criterion,device):

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images,labels in test_loader:
            images = images.to(device if torch.cuda.is_available() else "cpu")
            labels = labels.to(device if torch.cuda.is_available() else "cpu")
            
            outputs = model(images)
            loss = criterion(outputs,labels)
            total_loss += loss.item()

            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100*correct/total
    return {
        "loss":total_loss/len(test_loader),
        "accuracy":accuracy
    }