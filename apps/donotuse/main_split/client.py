import copy

import torch
from torch import nn
from torch.optim import Adam

from src.data.data_container import DataContainer
from src.federated.components.trainers import TorchTrainer
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams

t_optimizer = None


class Trainer(TorchTrainer):
    def __init__(self, initial_model):
        super().__init__()
        self.model = initial_model
        self.optimizer = Adam(self.model.parameters(), lr=0.01)
        self.device = torch.device('cuda')
        self.model_output = None
        self.is_trained = False

    def train(self, model: nn.Module, train_data: DataContainer, context: FederatedLearning.Context,
              config: TrainerParams):
        inputs, labels = (train_data.x, train_data.y) if isinstance(train_data, DataContainer) \
            else (train_data[0], train_data[1])
        inputs = inputs.to(self.device)
        labels = labels.clone().detach().long().to(self.device)
        self.model.to(self.device)
        self.optimizer.zero_grad()
        self.model_output = self.model(inputs)
        out = self.model_output.clone().detach().requires_grad_(True)
        self.is_trained = True
        return out, labels

    def model_copy(self):
        return copy.deepcopy(self.model)

    def backward(self, grad):
        self.model_output.backward(grad)
        self.optimizer.step()
        self.is_trained = False


class Client(Trainer):
    def __init__(self, data, initial_model):
        super().__init__(initial_model)
        self.data = data

    def local_train(self):
        return super(Client, self).train(None, self.data, None, None)
