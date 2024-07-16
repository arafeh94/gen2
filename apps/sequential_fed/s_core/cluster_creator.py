import copy

from sklearn.cluster import KMeans

from src.apis import federated_tools, utils
from src.apis.utils import group_dict_by_values


class ClusterCreator:
    def __init__(self, train_clients, model):
        self.model = model
        self.train_clients = train_clients
        self.trained_clients = {}
        self.is_trained = False
        self.compressed_weights = None

    def train(self, epochs=1, lr=0.0001):
        for client_idx, data in self.train_clients.items():
            model = copy.deepcopy(self.model)
            federated_tools.train(model, data.batch(), epochs=epochs, lr=lr)
            self.trained_clients[client_idx] = model
        self.is_trained = True

    def cluster(self, cluster_size, group_results=True):
        self.pre_check()
        clustered = {}
        self.compressed_weights = utils.pca(self.trained_clients)
        client_labels = KMeans(n_clusters=cluster_size).fit_predict(list(self.compressed_weights.values()))
        for client_id, label in zip(self.compressed_weights.keys(), client_labels):
            clustered[client_id] = label
        return group_dict_by_values(clustered) if group_results else clustered

    def pre_check(self):
        if not self.is_trained:
            self.is_trained = True
            self.train()


def build_clusters(train_clients, model, cluster_size, epochs, lr):
    cluster_creator = ClusterCreator(train_clients, model)
    cluster_creator.train(epochs, lr)
    clustered = cluster_creator.cluster(cluster_size)
    # check_results(train_clients, clustered)
    return clustered


def check_results(train_clients, cluster_results):
    client_labels = {}
    for client_id, data in train_clients.items():
        client_labels[client_id] = data.labels()
    original_grouped = group_dict_by_values(client_labels)
    cluster_grouped = group_dict_by_values(cluster_results)
    return original_grouped, cluster_grouped
