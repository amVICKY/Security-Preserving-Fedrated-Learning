import torch
import torch.nn as nn
import torch.optim as optim

from training.trainer import train
from training.evaluator import evaluate

def local_train(model,train_loader,test_loader,device):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )
    train_loss, train_acc = train(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )
    test_loss,test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    return model, {
        "train_loss":train_loss,
        "train_accuracy":train_acc,
        "test_loss":test_loss,
        "test_accuracy":test_acc
    }