import cProfile
import copy
import os
import pickle
import random

from apps.splitfed.core.client import Client
from apps.splitfed.core.clusters import generate_resources, Cluster
from apps.splitfed.core.splitfed import SplitFed
from apps.splitfed.core import clusters, dist
from apps.splitfed.models import MnistServer, MnistClient
from src.apis import lambdas
from src.apis.extensions import Dict
from src.data.data_distributor import ShardDistributor, ClusterDistributor
from src.data.data_loader import preload
from src.federated.subscribers.sqlite_logger import SQLiteLogger

tag = 'asd+' + str(random.randint(0, 9999999))
rounds = 50
test_data = preload('mnist10k').as_tensor()
client_model = MnistClient(784, 32, 10)
server_model = MnistServer(784, 32, 10)
clients_data = preload('mnist', ShardDistributor(150, 2), tag='12az3')
clients_data = clients_data.map(lambda k, dc: dc.shuffle(45)).map(lambdas.as_tensor)

split_clients = {}
for cid, cdt in clients_data.items():
    split_clients[cid] = Cluster(client_model, [Client.generate(cdt, copy.deepcopy(client_model))])

# split_clients = Dict(pickle.load(open('fast.pkl', 'rb')))
split_fed = SplitFed(server_model, client_model, split_clients, test_data, epochs=1, lr=0.01)

for i in range(rounds):
    split_fed.rand_resources()
    stats = split_fed.one_round()
    print(stats)
