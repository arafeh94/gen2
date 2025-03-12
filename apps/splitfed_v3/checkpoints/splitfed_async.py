import copy
import random
from collections import defaultdict

from matplotlib import pyplot as plt

from apps.splitfed.core.clusters import generate_resources
from apps.splitfed_v2.core.splitlearn import one_round
from src.apis.federated_tools import aggregate
from src.data.data_distributor import ShardDistributor, ClusterDistributor
from src.data.data_loader import preload
from apps.donotuse.split_learning import funcs
from apps.donotuse.split_learning import dist, models, clusters
from apps.donotuse.split_learning.server import Server

tag = 'asd+' + str(random.randint(0, 9999999))

rounds = 100
client_model = models.MnistClient(784, 32, 10)
server_model = models.MnistServer(784, 32, 10)

outer_distributor = ShardDistributor(10000, 2)
# 250 clients = cluster_client_num * clusters = 5 * 50
clients_clusters = preload('mnist', ClusterDistributor(outer_distributor, 20, 270, 270), tag=tag)
resource_clients = generate_resources(clients_clusters, copy.deepcopy(server_model), copy.deepcopy(client_model), 10,
                                      [1, 2])
test_data = preload('mnist10k').as_tensor()

server = Server(server_model, copy.deepcopy(client_model), test_data)

clusters_run = [1, 2, 2]
for i in range(rounds):
    clusters_run_index = 0
    for clsr in clusters_run:
        run_clusters = []
        for rs_key, rses in resource_clients.items():
            run_clusters.append(rses[clusters_run[clusters_run_index]])
        print(one_round(run_clusters, server))
        clusters_run_index += 1
