import random

from src.apis import lambdas, transformers
from src.apis.federated_tools import aggregate
from src.apis.rw import IODict
from src.data.data_loader import preload
from apps.main_split import models, dist
from apps.donotuse.main_split import funcs
from apps.main_split import clusters
from apps.main_split.server import Server

cache = IODict(f'cache/.tr.iod')
# clients in clusters: how many clients exists in each cluster. i.e. the tt number of cluster is pre-defined by dist.
c_cluster = [20]
# maximum number of clients in each cluster, e.i if each cluster have 20 clients but n_cluster = 5 means that we would
# have the clusters divided in 4 clusters of 5 clients each. n_cluster=1 means normal split_learning
n_cluster = 5
# client ratio, the number of selected client per round from each cluster in split learning
clrs = [5]
# learn rate
lrs = [0.001]
for lr in lrs:
    for clr in clrs:
        for t in c_cluster:
            rounds = 1_000
            cluster_size = t
            client_model = models.CifarClient()
            server_model = models.CifarServer()
            clients_data = preload('cifar10', dist.clustered(cluster_size), transformer=transformers.cifar10_rgb,
                                   tag=f'cluster_cifar{cluster_size}').map(lambda i, dc: dc.as_tensor())
            train_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[0]).map(lambdas.as_tensor)
            test_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[1]).reduce(lambdas.dict2dc).as_tensor()

            # federated learning
            # fed = FastFed(data=clients_data, rounds=rounds, client_ratio=0.1, model=models.CifarModel()).start()

            # split learning
            client_clusters = clusters.from_clients(train_data, client_model, n_cluster)
            as_list = list(client_clusters.items())
            random.shuffle(as_list)
            client_clusters = dict(as_list)
            server = Server(server_model, client_model, test_data)
            # configs
            split_accs = []
            for r in range(rounds):
                for cluster_index, client_cluster in client_clusters.items():
                    client_cluster.update_model(server.client_model)
                    selected_clients = random.choices(client_cluster.clients, k=min(clr, t))
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

            # fed_accs = fed.context.get_field('acc')

            # plt.grid()
            # p1 = plt.plot(fed_accs, '-.', label='Federated', linewidth=5)
            # p2 = plt.plot(split_accs, '-', label='Split', linewidth=5)
            cache.write(f'test_{t}_{clr}_{lr}', {'split': split_accs})
            # plt.legend()
            # plt.savefig(f'test_{t}r.png', bbox_inches='tight')
            # plt.clf()
            # plt.show()
