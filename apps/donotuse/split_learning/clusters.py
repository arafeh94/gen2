import copy
from typing import List, Dict

from apps.splitfed.core.client import Client


# from apps.donotuse.split_learning.client import Client


class Cluster:

    def __init__(self, initial_model, clients: List[Client]):
        self.model = initial_model
        self.clients = clients

    def update_model(self, new_model):
        self.model.load_state_dict(new_model.state_dict())
        for client in self.clients:
            client.model.load_state_dict(new_model.state_dict())


def from_clients(clients_data, initial_model, size_cluster=1) -> List[Cluster]:
    clusters = []
    for i in range(0, len(clients_data), size_cluster):
        cluster_data = clients_data.select(range(i, i + size_cluster))
        cluster_clients = [Client(cluster_data[c], copy.deepcopy(initial_model)) for c in cluster_data]
        clusters.append(Cluster(copy.deepcopy(initial_model), cluster_clients))
    return clusters
