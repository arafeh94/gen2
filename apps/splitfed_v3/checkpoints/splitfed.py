import copy
import random
from matplotlib import pyplot as plt

from apps.splitfed_v2.core.splitlearn import one_round
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from apps.donotuse.split_learning import dist, models, clusters
from apps.donotuse.split_learning.server import Server

rounds = 100
client_model = models.MnistClient(784, 32, 10)
server_model = models.MnistServer(784, 32, 10)
train_data = preload('mnist', ShardDistributor(150, 2), tag='12az3')
test_data = preload('mnist10k').as_tensor()
client_clusters = clusters.from_clients(train_data, client_model, 29)
random.shuffle(client_clusters)
server = Server(server_model, copy.deepcopy(client_model), test_data)
# configs
split_accs = []

for i in range(rounds):
    print(one_round(client_clusters, server))
