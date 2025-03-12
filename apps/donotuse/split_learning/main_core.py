import copy
import random

from matplotlib import pyplot as plt

from src.apis.federated_tools import aggregate
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from apps.donotuse.split_learning import funcs
from apps.donotuse.split_learning import models, clusters
from apps.donotuse.split_learning.server import Server


rounds = 100
client_model = models.MnistClient(784, 32, 10)
server_model = models.MnistServer(784, 32, 10)
train_data = preload('mnist', ShardDistributor(150, 2), tag='12az3')
test_data = preload('mnist10k').as_tensor()

# split learning
client_clusters = clusters.from_clients(train_data, client_model, 1)
as_list = list(client_clusters.items())
random.shuffle(as_list)
client_clusters = dict(as_list)
server = Server(server_model, copy.deepcopy(client_model), test_data)
# configs
split_accs = []
for r in range(rounds):
    for cluster_index, client_cluster in client_clusters.items():
        client_cluster.update_model(server.client_model)
        for client in client_cluster.clients:
            out, labels = client.local_train()
            grad = server.train(out, labels)
            client.backward(grad)
        weights = funcs.as_dict([c.model.state_dict() for c in client_cluster.clients])
        avg_weights = aggregate(weights, {})
        client_cluster.model.load_state_dict(avg_weights)
        server.client_model.load_state_dict(avg_weights)
    split_accs.append(server.infer())
    print(f'global_test_{r}', split_accs[-1])

plt.grid()
p2 = plt.plot(split_accs, '-', label='Split', linewidth=5)
plt.legend()
plt.show()
