import torch
from torch import nn
from torch.optim import Adam

from apps.donotuse.split_learning import funcs


class Server:
    def __init__(self, server_model, client_model, test_data, lr=0.0001):
        self.server_model = server_model
        self.client_model = client_model
        self.optimizer = Adam(self.server_model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.test_data = test_data
        self.device = torch.device('cuda')

    def train(self, client_output, labels):
        self.server_model.to(self.device)
        self.client_model.to(self.device)
        self.optimizer.zero_grad()
        output = self.server_model(client_output)
        loss = self.criterion(output, labels)
        loss.backward()
        grad = client_output.grad.clone().detach()
        self.optimizer.step()
        return grad

    def infer(self):
        return funcs.infer(self.server_model, self.client_model, self.test_data)
