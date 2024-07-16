import copy
import os
import pickle
import random

from apps.splitfed.core.clusters import generate_resources, Cluster
from apps.splitfed.core.splitfed import SplitFed
from apps.splitfed.core import clusters, dist
from apps.splitfed.models import MnistServer, MnistClient
from src.apis import lambdas
from src.data.data_distributor import ShardDistributor, ClusterDistributor
from src.data.data_loader import preload
from src.federated.subscribers.sqlite_logger import SQLiteLogger

tag = 'asd+' + str(random.randint(0, 9999999))
rounds = 50
test_data = preload('mnist10k').as_tensor()
client_model = MnistClient(784, 32, 10)
server_model = MnistServer(784, 32, 10)
# 5 clusters = total_labels/shards = 10/2
outer_distributor = ShardDistributor(10000, 2)
# 250 clients = cluster_client_num * clusters = 5 * 50
clients_clusters = preload('mnist', ClusterDistributor(outer_distributor, 20, 270, 270), tag=tag)
resource_clients = generate_resources(clients_clusters, copy.deepcopy(server_model), copy.deepcopy(client_model), 20,
                                      [1, 2])
split_clients = {}
for out_cluster_key, out_cluster in resource_clients.items():
    for in_cluster_key, in_cluster in out_cluster.items():
        for client in in_cluster.clients:
            split_clients[client.cid] = Cluster(copy.deepcopy(client_model), [client])

split_fed = SplitFed(server_model, client_model, split_clients, test_data,
                     epochs=1, lr=0.001, fed_speed=0)

for i in range(rounds):
    split_fed.rand_resources()
    stats = split_fed.one_round()
    print(stats)
