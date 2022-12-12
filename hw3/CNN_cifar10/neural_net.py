import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.05), 
        )

        self.flatten = nn.Flatten()

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(7200, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv_layer(x)
        
        x = self.flatten(x)

        x = self.fc_layer(x) 
        
        return x

    def cal_loss(self, pred, target):
        loss = self.criterion(pred, target)
        return loss
