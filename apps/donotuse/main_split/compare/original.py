import copy

from src.apis import lambdas
from src.apis.rw import IODict
from src.data.data_loader import preload
from apps.main_split import models, dist
from apps.main_split import clusters
from apps.main_split.server import Server

results = IODict('./res')

client_cluster_sizes = [10]

for client_cluster_size in client_cluster_sizes:
    rounds = 100
    client_model = models.MnistClient(784, 32, 10)
    server_model = models.MnistServer(784, 32, 10)
    clients_data = preload('mnist', dist.clustered(client_cluster_size, 300), tag=f'cluster{client_cluster_size}p{300}')
    train_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[0]).map(lambdas.as_tensor)
    test_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[1]).reduce(lambdas.dict2dc).as_tensor()

    # split learning
    client_clusters = clusters.from_clients(train_data, copy.deepcopy(client_model), 10)
    fed_server = Server(server_model)
    client_model = copy.deepcopy(client_model)
    # configs
    split_accs = []
    for r in range(rounds):
        for cluster_index, client_cluster in client_clusters.items():
            for client in client_cluster.clients:
                client.model.load_state_dict(client_model.state_dict())
                for e in range(1):
                    out, labels = client.local_train()
                    grad = fed_server.train(out, labels)
                    client.backward(grad)
                client_model.load_state_dict(client.model.state_dict())
        split_accs.append(fed_server.infer(client_model, test_data))
        print(f'global_test_{r}', split_accs[-1])
    results.write('original_split', split_accs)
