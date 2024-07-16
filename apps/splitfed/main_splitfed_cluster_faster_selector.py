import copy
import os
from collections import defaultdict
from apps.splitfed.core.clusters import generate_resources
from apps.splitfed.core.splitfed import SplitFed, ClusterExecutor
from apps.splitfed.models import MnistClient, MnistServer
from src.data.data_distributor import ClusterDistributor, ShardDistributor
from src.data.data_loader import preload
from src.federated.subscribers.sqlite_logger import SQLiteLogger

logger = SQLiteLogger(os.path.splitext(os.path.basename(__file__))[0], 'logs.db')

rounds = 50

# init data
client_model = MnistClient(784, 32, 10)
server_model = MnistServer(784, 32, 10)

train_data, test_data = preload('mnist').split(0.9)
test_data = test_data.as_tensor()
outer_distributor = ShardDistributor(10000, 2)
clients_clusters = preload('mnist', ClusterDistributor(outer_distributor, 10, 500, 500))
clusters = generate_resources(clients_clusters, copy.deepcopy(server_model), copy.deepcopy(client_model), 5,
                              [2, 1])

resources_based = defaultdict(dict)
split_feds: dict[int, SplitFed] = {}
for key_out, out in clusters.items():
    for key_in, inc in out.items():
        resources_based[key_in][key_out] = inc
for key_rb, rb in resources_based.items():
    split_fed = SplitFed(copy.deepcopy(server_model), copy.deepcopy(client_model), rb, test_data, epochs=10, lr=0.1,
                         fed_speed=key_rb)
    split_feds[split_fed.fed_speed] = split_fed

executor = ClusterExecutor([sf for sf in split_feds.values()])

for i in range(50):
    executor.one_round()
    print(executor.infer())
print('asd')
