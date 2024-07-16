from typing import List, Dict

from apps.main_split.client import Client


class Cluster:

    def __init__(self, initial_model, clients: List[Client]):
        self.model = initial_model
        self.clients = clients

    def update_model(self, new_model):
        self.model.load_state_dict(new_model.state_dict())
        for client in self.clients:
            client.model.load_state_dict(new_model.state_dict())


def from_clients(clients_data, initial_model, size_cluster=1) -> Dict[int, Cluster]:
    clusters = {}
    cluster_id = 0
    for i in range(0, len(clients_data), size_cluster):
        cluster_data = clients_data.select(range(i, i + size_cluster))
        cluster_clients = [Client(cluster_data[c], initial_model) for c in cluster_data]
        clusters[cluster_id] = Cluster(initial_model, cluster_clients)
        cluster_id += 1
    return clusters
