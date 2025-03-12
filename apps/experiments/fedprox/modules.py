import copy
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import Parameter
from torch.optim import Optimizer
from tqdm import tqdm

from src.apis import federated_tools
from src.apis.extensions import Dict
from src.data.data_container import DataContainer
from src.federated.federated import FederatedLearning
from src.federated.protocols import Trainer, TrainerParams


class FedProxTrainer(Trainer):
    def __init__(self):
        self.device = torch.device('cpu')

    def train(self, model: nn.Module, train_data: DataContainer, context, config: TrainerParams) -> Tuple[any, int]:
        model.to(self.device)
        model.train()
        optimizer = config.get_optimizer()(model)
        criterion = nn.CrossEntropyLoss()
        global_param = copy.deepcopy(context.model).to(self.device).parameters()

        epoch_loss = []
        epochs = range(config.epochs)
        if 'verbose' in config.args and config.args['verbose']:
            epochs = tqdm(epochs)
        for _ in epochs:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data.batch(config.batch_size)):
                x = x.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)

                proximal_term = 0.0
                for param, g_param in zip(model.parameters(), global_param):
                    proximal_term += torch.norm(param - g_param) ** 2
                loss += (config.args['mu'] / 2) * proximal_term

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        weights = model.cpu().state_dict()
        return weights, len(train_data)
