import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    
    def __init__(self,num_classes=10):
        super(CNN,self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(64*7*7,128)
        self.fc2 = nn.Linear(128,num_classes)

    def __str__(self):
        return "This is the simple Vision Model"

    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0),-1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x