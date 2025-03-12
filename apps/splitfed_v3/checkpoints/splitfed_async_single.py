import copy
import random
from collections import defaultdict
from sklearn.cluster import KMeans

from apps.splitfed_v2.core.client import Client
from apps.splitfed_v2.core.clusters import Cluster
from apps.splitfed_v2.core.server import Server
from apps.splitfed_v2.core.splitlearn import one_round
from src.apis.extensions import CycleList
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from apps.donotuse.split_learning import models

tag = 'asd+' + str(random.randint(0, 9999999))

rounds = 200
client_model = models.MnistClient(784, 32, 10)
server_model = models.MnistServer(784, 32, 10)

speeds = [1, 2]

mnist = preload('mnist', ShardDistributor(300, 2))
test_data = preload('mnist10k').as_tensor()


def cluster(clients, out_cluster_number: int, cl_speeds: list, model, speed_weights=None):
    clients_ys = [dt.labels() for cid, dt in clients.items()]
    kmeans = KMeans(n_clusters=out_cluster_number, random_state=42)
    kmeans.fit(clients_ys)
    clustered = defaultdict(lambda: defaultdict(lambda: Cluster(copy.deepcopy(client_model), [])))
    for cid, client_dc in clients.items():
        cluster_id = kmeans.predict([client_dc.labels()])[0]
        client_speed = random.choices(cl_speeds, weights=speed_weights, k=1)[0]
        client = Client.generate(client_dc.as_tensor(), copy.deepcopy(model), client_speed)
        clustered[cluster_id][client.speed].append(client)
    return clustered


res = cluster(mnist, 5, [1, 2], client_model)

server = Server(server_model, copy.deepcopy(client_model), test_data)
clusters_run = [1, 2, 2, 2]
for i in range(rounds):
    clusters_run_index = 0
    for clsr in clusters_run:
        run_clusters = []
        for rs_key, rses in res.items():
            run_clusters.append(rses[clusters_run[clusters_run_index]])
        print(one_round(run_clusters, server))
        clusters_run_index += 1
