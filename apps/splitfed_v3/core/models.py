import torch
import torch.nn as nn
import torch.optim as optim


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 3 input features, 64 hidden units
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)  # 64 hidden units, 32 hidden units
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)  # 32 hidden units, 1 output (regression value)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
