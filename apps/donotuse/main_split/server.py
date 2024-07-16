import copy

import torch
from torch import nn
from torch.optim import Adam

from apps.donotuse.main_split import funcs


class Server:
    def __init__(self, model, lr=0.0001):
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda')

    def train(self, client_output, labels):
        self.model.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(client_output)
        loss = self.criterion(output, labels)
        loss.backward()
        grad = client_output.grad.clone().detach()
        self.optimizer.step()
        return grad

    def model_copy(self):
        return copy.deepcopy(self.model)

    def infer(self, client_model, test_data):
        return funcs.infer(self.model_copy(), client_model, test_data)
