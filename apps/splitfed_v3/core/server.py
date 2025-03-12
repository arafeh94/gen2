import copy

import torch
from torch import nn
from torch.optim import Adam

from apps.donotuse.split_learning import funcs, models


class Server:
    def __init__(self, server_model, client_model, test_data, lr=0.000001):
        self.server_model = server_model
        self.client_model = client_model
        self.optimizer = Adam(self.server_model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.test_data = test_data
        self.device = torch.device('cuda')
        self.lr = lr

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

    def models(self):
        return self.server_model.state_dict(), self.client_model.state_dict()

    def infer(self):
        return funcs.infer(self.server_model, self.client_model, self.test_data)

    def copy(self):
        server_model = copy.deepcopy(self.server_model)
        client_model = copy.deepcopy(self.client_model)
        return Server(server_model, client_model, self.test_data, lr=self.lr)

    def __deepcopy__(self, memo):
        new_instance = self.__class__(
            server_model=copy.deepcopy(self.server_model, memo),
            client_model=copy.deepcopy(self.client_model, memo),
            test_data=copy.deepcopy(self.test_data, memo),
            lr=self.optimizer.param_groups[0]['lr']
        )
        new_instance.optimizer = copy.deepcopy(self.optimizer, memo)
        new_instance.criterion = self.criterion
        new_instance.device = self.device
        return new_instance
