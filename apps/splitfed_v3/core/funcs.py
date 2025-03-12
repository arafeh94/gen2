import copy

import torch
from torch import nn

from apps.splitfed_v2.core.client import Client
from src.apis import lambdas
from src.apis.utils import TimeCheckpoint
from src.data import data_loader
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from src.federated.events import FederatedSubscriber
from src.federated.federated import FederatedLearning


def as_dict(items: list):
    d = {}
    for index, item in enumerate(items):
        d[index] = item
    return d


def average_weights(w, datasize):
    """
    Returns the average of the weights.
    """

    for i, data in enumerate(datasize):
        for key in w[i].keys():
            w[i][key] *= float(data)

    w_avg = copy.deepcopy(w[0])

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

    return w_avg


def get_test_data():
    return preload('mnist10k').as_tensor()


def infer(server_model, client_model, data):
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


class SQLoggerCustom(FederatedSubscriber):
    def __init__(self, logger, clients: [Client]):
        super().__init__()
        self.logger = logger
        self.selected_clients = []
        self.clients = clients

    def on_round_start(self, params):
        for client in self.clients:
            client.randomize_resources()

    def on_trainers_selected(self, params):
        self.selected_clients = params['trainers_ids']

    def on_round_end(self, params):
        context: FederatedLearning.Context = params['context']
        last_record: dict = context.history[context.round_id]
        round_times = []
        for selected_client in self.selected_clients:
            round_times.append(self.clients[selected_client].exec_time())
        round_time = max(round_times)
        self.logger.log(context.round_id, acc=last_record['acc'], loss=last_record['loss'],
                        round_time=round_time, cluster_time=round_times, clients_time=round_times,
                        cluster_selection_size=1, speed=1, round_num=context.round_id, iter_num=1)
