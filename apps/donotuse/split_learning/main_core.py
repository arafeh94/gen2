import copy
import random

from matplotlib import pyplot as plt

from src.apis import lambdas
from src.apis.federated_tools import aggregate
from src.app.prebuilt import FastFed
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from apps.donotuse.split_learning import funcs
from apps.donotuse.split_learning import dist, models, clusters
from apps.donotuse.split_learning.server import Server

test_sizes = [20]

for t in test_sizes:
    rounds = 100
    cluster_size = t
    client_model = models.MnistClient(784, 32, 10)
    server_model = models.MnistServer(784, 32, 10)
    # clients_data = preload('mnist', dist.clustered(cluster_size), tag=f'cluster{cluster_size}')
    # train_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[0]).map(lambdas.as_tensor)
    # test_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[1]).reduce(lambdas.dict2dc).as_tensor()
    train_data = preload('mnist', ShardDistributor(150, 2), tag='12az3')
    test_data = preload('mnist10k').as_tensor()
    # federated learning
    # fed = FastFed(data=train_data, rounds=rounds, client_ratio=0.3).start()

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

    fed_accs = fed.context.get_field('acc')

    plt.grid()
    p1 = plt.plot(fed_accs, '-.', label='Federated', linewidth=5)
    p2 = plt.plot(split_accs, '-', label='Split', linewidth=5)
    plt.legend()
    # plt.savefig(f'test_{t}r.png', bbox_inches='tight')
    # plt.clf()
    plt.show()
