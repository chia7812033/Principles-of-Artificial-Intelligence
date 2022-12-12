import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc5(out)
        out = self.softmax(out)
        return out

    def cal_loss(self, pred, target):
        loss = self.criterion(pred, target)
        return loss
