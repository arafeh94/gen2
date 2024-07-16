import copy
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict

from sklearn.cluster import KMeans

from apps.donotuse.main_split import funcs
from apps.splitfed_v2.core import cluster_creator
from apps.splitfed_v2.core.client import Client
from apps.splitfed_v2.core.clusters import Cluster
from apps.splitfed_v2.core.selector import client_selection, quality
from apps.splitfed_v2.core.server import Server
from src.apis import utils
from src.apis.extensions import LinkedList, Node, CycleList
from src.apis.federated_tools import aggregate, asyncgregate


def one_round(cls, srvr):
    random.shuffle(cls)
    for client_cluster in cls:
        client_cluster.update_model(srvr.client_model)
        for client in client_cluster.get_trainers():
            out, labels = client.local_train()
            grad = srvr.train(out, labels)
            client.backward(grad)
        weights = funcs.as_dict([c.model.state_dict() for c in client_cluster.clients])
        cls_samples = funcs.as_dict([len(c) for c in client_cluster.clients])
        avg_weights = aggregate(weights, cls_samples)
        # client_cluster.model.load_state_dict(avg_weights)
        srvr.client_model.load_state_dict(avg_weights)
    return srvr.infer()


def one_round_async(cls, srvr: Server):
    srvr.client_model = srvr.client_model.to('cuda')
    srvr.server_model = srvr.server_model.to('cuda')
    random.shuffle(cls)
    for client_cluster in cls:
        client_cluster.update_model(srvr.client_model)
        train_servers = []
        for client in client_cluster.get_trainers():
            train_srvr = srvr.copy()
            train_servers.append(train_srvr)
            out, labels = client.local_train()
            grad = train_srvr.train(out, labels)
            client.backward(grad)
        cls_weights = funcs.as_dict([c.model.state_dict() for c in client_cluster.clients])
        cls_samples = funcs.as_dict([len(c) for c in client_cluster.clients])
        srv_weights = funcs.as_dict([c.server_model.state_dict() for c in train_servers])
        avg_cls_weights = aggregate(cls_weights, cls_samples)
        avg_srv_weights = aggregate(srv_weights, cls_samples)
        client_cluster.model.load_state_dict(avg_cls_weights)
        srvr.client_model.load_state_dict(avg_cls_weights)
        srvr.server_model.load_state_dict(avg_srv_weights)
    return srvr.infer()


def one_round_resource(cls, srvr, is_parallel=True, is_selection=True, bad_ratio=.05):
    clusters_time = []
    cluster_clients_time = []
    cluster_clients_size = []
    for client_cluster in cls:
        # cls_len = len(client_cluster.clients)
        # performance = PerformanceArrays.n_bad(cls_len, math.ceil(cls_len * .25))
        for idx, client in enumerate(client_cluster.clients):
            client.randomize_resources(upto=PerformanceArrays.bad_roulette(bad_ratio))
    cls = client_selection(cls) if is_selection else cls
    for client_cluster in cls:
        clients_time = {}
        for client in client_cluster.clients:
            if client.is_trainable:
                clients_time[client.cid] = client.exec_time()
        time_taken = max(clients_time.values()) if is_parallel else sum(clients_time.values())
        clusters_time.append(time_taken)
        cluster_clients_time.append(clients_time)
        cluster_clients_size.append(len(clients_time))
    accuracy, loss = one_round_async(cls, srvr) if is_parallel else one_round(cls, srvr)
    # accuracy, loss = one_round(cls, srvr)
    return {'acc': accuracy, 'loss': loss, 'round_time': sum(clusters_time), 'clusters_time': clusters_time,
            'clients_time': str(cluster_clients_time), 'cluster_selection_size': str(cluster_clients_size)}


# def DoubleClusteredSplitFed(clients):
#     one_epoch_training(clients.models)
#     update_resource_capacities(clients)
# 	out_clusters = iid_clustering(clients.models)
# 	for out_cluster, clients in out_clusters:
# 		out_cluster.in_clusters = resource_clustering(clients.resources)
# 	for i in rounds:
# 		out_cluster = shuffle_select_next_cluster()
# 		while out_cluster:
# 			for in_cluster in out_cluster.in_clusters: \\In parallel
# 				update_available_resources(in_cluster.clients)
# 				selected_clients = selection(in_cluster.clients)
# 				for client, server in selected_clients: \\In parallel
# 					client.feed_forward()
# 					server.feed_forward()
# 					server.backpropagate()
# 					client.backpropage()
# 					client.optimizer.step()
# 				async_aggregate(out_cluster.model, in_cluster.clients)
# 			out_cluster = shuffle_select_next_cluster()
def crossgregate(advanced, late, staled_round):
    """
    Args:
        advanced: (server, client)
        late: (server, client)
        staled_round: how much the second is late

    Returns: (server,client) asynced weights

    """
    server_weights = asyncgregate(advanced[0], late[0], staled_round)
    client_weights = asyncgregate(advanced[1], late[1], staled_round)
    return server_weights, client_weights


def cluster_label(clients, out_cluster_number: int, cl_speeds: list, model, lr, speed_weights=None):
    clients_ys = [dt.as_list().labels() for cid, dt in clients.items()]
    kmeans = KMeans(n_clusters=out_cluster_number, random_state=42)
    kmeans.fit(clients_ys)
    clustered = defaultdict(lambda: defaultdict(lambda: Cluster(copy.deepcopy(model), [])))
    for cid, client_dc in clients.items():
        cluster_id = kmeans.predict([client_dc.as_list().labels()])[0]
        client_speed = random.choices(cl_speeds, weights=speed_weights, k=1)[0]
        client = Client.generate(client_dc.as_tensor(), copy.deepcopy(model), client_speed, lr=lr)
        clustered[cluster_id][client.speed].append(client)
    return dict(clustered)


def cluster(clients, cl_speeds: list, model, lr, speed_weights=None, predictor=None):
    clustered = defaultdict(lambda: defaultdict(lambda: Cluster(copy.deepcopy(model), [])))
    for cid, client_dc in clients.items():
        cluster_id = predictor(cid)
        client_speed = random.choices(cl_speeds, weights=speed_weights, k=1)[0]
        client = Client.generate(client_dc.as_tensor(), copy.deepcopy(model), client_speed, lr=lr)
        clustered[cluster_id][client.speed].append(client)
    return dict(clustered)


def as_clients(clients_data, model, cl_speeds: list, speed_weights=None):
    clients = []
    for cid, client_dc in clients_data.items():
        client_speed = random.choices(cl_speeds, weights=speed_weights, k=1)[0]
        client = Client.generate(client_dc.as_tensor(), copy.deepcopy(model), client_speed)
        clients.append(client)
    return clients


class ClsIterator:
    def __init__(self, speeds: list[float], asyncro=True):
        self.speeds = sorted(speeds)
        self.linked = LinkedList.create(self.speeds)
        self.current: Node or None = None
        self.node_history = defaultdict(list)
        self.count = -1
        self.asyncro = asyncro

    def next(self):
        if self.asyncro:
            return self.next_async()
        return self.next_sync()

    def next_async(self):
        self.count += 1
        cache = (None, None)
        if self.current is None:
            self.current = self.linked.head
            return *cache, 1
        if self.current.has_prev() and self.is_time_exceeded():
            cache = self.current.value, self.current.prev_node().value
            self.current = self.current.prev_node()
            should_stop = self.is_time_exceeded() if self.current.has_prev() else 0
            return *cache, self.current.has_prev(), should_stop
        elif self.current.has_next():
            self.current = self.current.next_node()
        return *cache, 1

    def next_sync(self):
        self.count += 1
        cache = (None, None)
        if self.current is None:
            self.current = self.linked.head
            return *cache, 1
        self.current = self.current.next_node()
        should_continue = self.current is not None
        return *cache, should_continue

    def val(self):
        return self.current.value

    def append_time(self, exec_time):
        self.node_history[self.current.value].append(exec_time)

    def is_time_exceeded(self):
        current_history = self.node_history[self.current.value]
        prev_history = self.node_history[self.current.prev_node().value]
        return sum(current_history) > sum(prev_history)

    def measure_delay(self, delayed):
        latest_delay = float(self.node_history[delayed][-1])
        cursor = 0
        while latest_delay > 0 and abs(cursor) < len(self.node_history[self.speeds[-1]]):
            latest_advanced = self.node_history[self.speeds[-1]][cursor - 1]
            latest_delay -= latest_advanced
            cursor -= 1
        return abs(cursor)

    def counter(self):
        return self.count


class PerformanceArrays:
    @staticmethod
    def n_bad(ln, n):
        performance = [0.9] * ln
        performance[:n] = [0.3] * n
        return performance

    @staticmethod
    def rand(ln):
        return [random.uniform(0.01, 0.99995) for _ in range(ln)]

    @staticmethod
    def default(ln, mx=0.95):
        return [mx] * ln

    @staticmethod
    def bad_roulette(bad_ratio=0.01, bad_val=0.1, good_val=0.95):
        is_bad = random.random() < bad_ratio
        return bad_val if is_bad else good_val


def get_clients(distributed_dataset, out_cluster_size, clusters_speeds, client_model, lr, id_model,
                dataset_path='./files/dt'):
    if os.path.exists(dataset_path):
        return pickle.load(open(dataset_path, 'rb'))
    else:
        clustered = cluster_creator.build_clusters(distributed_dataset, copy.deepcopy(id_model), out_cluster_size,
                                                   5, lr, inverse=True)

        double = cluster(distributed_dataset, clusters_speeds, client_model, lr=lr,
                         predictor=lambda cid: clustered[cid][0])
        double_clustered = utils.ddict2dict(double)
        pickle.dump(double_clustered, open(dataset_path, 'wb'))
        return double_clustered


def clients1d(double_clustered):
    clients = []
    for key, out_c in double_clustered.items():
        for _, inner in out_c.items():
            clients.extend(inner.clients)
    return clients
