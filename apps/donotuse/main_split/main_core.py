import copy
import random

from matplotlib import pyplot as plt

from src.apis import lambdas
from src.apis.federated_tools import aggregate
from src.data.data_loader import preload
from apps.main_split import models, dist
from apps.donotuse.main_split import funcs
from apps.main_split import clusters
from apps.main_split.server import Server

client_cluster_sizes = [1]

for client_cluster_size in client_cluster_sizes:
    rounds = 100
    client_model = models.MnistClient(784, 32, 10)
    server_model = models.MnistServer(784, 32, 10)
    clients_data = preload('mnist', dist.clustered(client_cluster_size, 300), tag=f'cluster{client_cluster_size}p{300}')
    train_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[0]).map(lambdas.as_tensor)
    test_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[1]).reduce(lambdas.dict2dc).as_tensor()

    # split learning
    client_clusters = clusters.from_clients(train_data, client_model, 4)
    as_list = list(client_clusters.items())
    random.shuffle(as_list)
    client_clusters = dict(as_list)
    fed_server = Server(server_model)
    client_handlers = []
    client_model = copy.deepcopy(client_model)
    # configs
    split_accs = []
    for r in range(rounds):
        for cluster_index, client_cluster in client_clusters.items():
            for client in client_cluster.clients:
                handler = Server(fed_server.model_copy())
                for e in range(1):
                    out, labels = client.local_train()
                    grad = handler.train(out, labels)
                    client.backward(grad)
                client_handlers.append(handler)
            weights_clients = funcs.as_dict([c.model_copy().state_dict() for c in client_cluster.clients])
            weights_servers = funcs.as_dict([s.model_copy().state_dict() for s in client_handlers])
            avg_weights_clients = aggregate(weights_clients, {})
            avg_weights_server = aggregate(weights_servers, {})
            client_model.load_state_dict(avg_weights_clients)
            fed_server.model.load_state_dict(avg_weights_server)
            client_handlers.clear()
        split_accs.append(fed_server.infer(client_model, test_data))
        print(f'global_test_{r}', split_accs[-1])


    plt.grid()
    p2 = plt.plot(split_accs, '-', label='Split', linewidth=5)
    plt.legend()
    plt.show()
