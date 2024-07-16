import copy
import random
from collections import defaultdict
from sklearn.cluster import KMeans

from apps.splitfed_v2.core import splitlearn
from apps.splitfed_v2.core.client import Client
from apps.splitfed_v2.core.clusters import Cluster
from apps.splitfed_v2.core.server import Server
from apps.splitfed_v2.core.splitlearn import one_round, ClsIterator
from src.apis.extensions import CycleList, Dict
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from apps.donotuse.split_learning import models

tag = 'asd+' + str(random.randint(0, 9999999))

rounds = 200
client_model = models.MnistClient(784, 32, 10)
server_model = models.MnistServer(784, 32, 10)

mnist = preload('mnist', ShardDistributor(300, 2))
test_data = preload('mnist10k').as_tensor()

cluster_speeds = [0.1, .25, .9]
out_clusters = 5


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


double_clustered = cluster(mnist, out_clusters, cluster_speeds, client_model)
servers = Dict()
for cluster_speed in cluster_speeds:
    servers[cluster_speed] = Server(copy.deepcopy(server_model), copy.deepcopy(client_model), test_data)
for i in range(rounds):
    itero = ClsIterator(cluster_speeds)
    while True:
        cross = itero.next()
        if isinstance(cross, tuple) and all(cross[:2]):
            print("aggregation: {}".format(cross))
            server_weights, clients_weights = splitlearn.crossgregate(
                servers[cross[0]].models(),
                servers[cross[1]].models(), 3)
            for speed_idx, server in servers.select([cross[0], cross[1]]).items():
                server.server_model.load_state_dict(server_weights)
                server.client_model.load_state_dict(clients_weights)
        if not cross[2]:
            break
        if len(cross) > 3 and cross[3]:
            continue
        speed_idx = itero.val()
        run_clusters = []
        for rs_key, outer in double_clustered.items():
            run_clusters.append(outer[speed_idx])
        stats = splitlearn.one_round_resource(run_clusters, servers[speed_idx])
        print("speed_idx - {}: {}".format(speed_idx, stats))
        itero.append_time(stats['tt'])
    print('round_end: {}'.format(i))
    print('------------------------------------')
