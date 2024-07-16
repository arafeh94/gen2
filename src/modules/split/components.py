import copy
from typing import Tuple

import torch
from torch import nn

from src.apis.extensions import Dict
from src.data.data_container import DataContainer
from src.federated.components.trainers import TorchTrainer
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams, Aggregator, ModelInfer


class SplitTrainer(TorchTrainer):
    def __init__(self, cut_layer):
        super().__init__()
        self.client_model = None
        self.server_model = None
        self.cut_layer = cut_layer

    def train(self, model: nn.Module, train_data: DataContainer, context: FederatedLearning.Context,
              config: TrainerParams) -> Tuple[any, int]:
        self.update_models(model)
        split_client = SplitClient(config.optimizer)
        out, labels = split_client.train(self.client_model, train_data)
        grad = self._train(config.optimizer, config.criterion, out, labels)
        split_client.backward(grad)
        return (self.client_model.state_dict(), self.server_model.state_dict()), len(train_data)

    def _train(self, optimizer, criterion, client_output, labels):
        self.server_model.to(self.device)
        optimizer.zero_grad()
        output = self.server_model(client_output)
        loss = criterion(output, labels)
        loss.backward()
        grad = client_output.grad.clone().detach()
        optimizer.step()
        return grad

    def update_models(self, model: nn.Module):
        self.client_model = nn.Sequential(*list(copy.deepcopy(model).children())[:self.cut_layer])
        self.server_model = nn.Sequential(*list(copy.deepcopy(model).children())[self.cut_layer:])


class SplitClient:
    def __init__(self, optimizer, device='cuda'):
        self.device = device
        self.model_output = None
        self.optimizer = optimizer

    def train(self, model, data):
        inputs, labels = (data.x, data.y)
        inputs = inputs.to(self.device)
        labels = labels.clone().detach().long().to(self.device)
        model.to(self.device)
        self.optimizer.zero_grad()
        self.model_output = model(inputs)
        out = self.model_output.clone().detach().requires_grad_(True)
        return out, labels

    def backward(self, grad):
        self.model_output.backward(grad)
        self.optimizer.step()


class SplitAggregator(Aggregator):
    def aggregate(self, trainers_models_weight_dict: Dict[int, nn.ModuleDict], sample_size: Dict[int, int],
                  round_id: int) -> nn.ModuleDict:
        # should return always one model
        pass


class SplitInfer(ModelInfer):
    def __init__(self, batch_size: int, criterion, device=None):
        super().__init__(batch_size, criterion)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def infer(self, model: nn.Module, test_data: DataContainer):
        device = torch.device('cuda')
        client_model = client_model.to(device)
        server_model = server_model.to(device)
        client_model.eval()
        server_model.eval()
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            corr_num = 0
            total_num = 0
            val_loss = 0.0
            val_x, val_label = data.x, data.y
            val_x = val_x.to(device)
            val_label = val_label.clone().detach().long().to(device)

            val_output = client_model(val_x)
            val_output = server_model(val_output)
            loss = criterion(val_output, val_label)
            val_loss += loss.item()
            model_label = val_output.argmax(dim=1)
            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)
            test_accuracy = corr_num / total_num
            test_loss = val_loss / val_label.size(0)
            return test_accuracy
