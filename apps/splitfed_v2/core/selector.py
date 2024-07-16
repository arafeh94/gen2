import json
import math
import os
import pickle
import time
import typing

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.gaussian_process import GaussianProcessRegressor

from apps.genetics.src import alg_genetic
from apps.genetics.src.alg_genetic import ClusterSelector
from apps.splitfed_v2.core.client import Client
from apps.splitfed_v2.core.clusters import Cluster
from apps.splitfed_v2.core.memetic import memetic_algorithm
from apps.splitfed_v2.core.models import RegressionModel
from src import manifest
from src.apis import utils


def client_selection(cls):
    reset_selection(cls)
    for cluster in cls:
        cluster: Cluster
        heuristic_selection(cluster)
    return cls


def reset_selection(cls):
    for cluster in cls:
        cluster: Cluster
        for client in cluster.clients:
            client.is_trainable = True


def deny_all(cls):
    for client in cls.clients:
        client.is_trainable = False


def heuristic_selection(one_cluster: Cluster):
    deny_all(one_cluster)
    fitness = SelectionFitness(one_cluster)

    slc = {}
    for client in one_cluster.clients:
        slc[client.cid] = [client]
    cluster_selector = ClusterSelector(slc)
    c_size = (min(2, len(one_cluster.clients)), len(one_cluster.clients))
    best, solutions = alg_genetic.ga(fitness.evaluate, cluster_selector, 99999999, 50, c_size=c_size, p_size=20)
    clients = fitness.map_selection2(best)
    for client in clients:
        client.is_trainable = True


def gp(clients):
    return max(ExecutionTimeModel().predict(clients))


def quality(clients: typing.List[Client], model):
    global_weights = utils.flatten_weights(model.state_dict())
    local_weights = [utils.flatten_weights(client.model.state_dict()) for client in clients]
    transformed = PCA().fit_transform(global_weights + local_weights)

    g_quality = utils.get_average_weight_divergence(transformed[0], transformed[1:])
    return g_quality


class ExecutionTimeModel:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ExecutionTimeModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.model: RegressionModel = pickle.load(open(self._path('nn_model.pkl'), 'rb'))
        self.model.eval()

    def _path(self, item):
        try1 = './files/{}'.format(item)
        if os.path.exists(try1):
            return try1
        paths = [item, 'files', './']
        path = ''
        for e_path in paths:
            path = '/' + e_path + path
            if os.path.exists(path):
                return path
        raise Exception("Can't find model")

    def _predict(self, resources: typing.List[typing.List]) -> typing.List[float]:
        with torch.no_grad():
            x = torch.FloatTensor(resources).to('cuda')
            predictions = self.model(x).cpu().numpy()
        return predictions

    def predict(self, clients: typing.List[Client]):
        resources = []
        for client in clients:
            resources.append([client.configs['ram_a'], client.configs['cpu_a'], client.configs['disk_a']])
        return self._predict(resources)


class SelectionFitness:
    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.selection = [0] * len(self.cluster.clients)
        self.evals = {}
        self.clients_pca = self._calculate_pca()
        self.clients_exc = self._calculate_exec()
        self._to_cuda()

    def _to_cuda(self):
        for client in self.cluster.clients:
            client.model.to('cuda')

    def _calculate_pca2(self):
        flattened_weights = [utils.flatten_weights(client.model.cpu().state_dict()) for client in self.cluster.clients]
        flattened_weights.append(utils.flatten_weights(self.cluster.model.cpu().state_dict()))
        pca = PCA()
        transformed = pca.fit_transform(flattened_weights)
        clients_pca = {}
        for idx, client in enumerate(self.cluster.clients):
            clients_pca[client.cid] = transformed[idx]
        clients_pca['global'] = transformed[-1]
        return clients_pca

    def _calculate_pca(self):
        flattened_weights = [utils.flatten_state_dict(client.model.state_dict()) for client in self.cluster.clients]
        flattened_weights.append(utils.flatten_state_dict(self.cluster.model.cuda().state_dict()))

        stacked_weights = torch.stack(flattened_weights).cuda()
        pca_result = torch.pca_lowrank(stacked_weights)
        transformed = pca_result[0]
        clients_pca = {}
        for idx, client in enumerate(self.cluster.clients):
            clients_pca[client.cid] = transformed[idx].cpu().numpy()

        clients_pca['global'] = transformed[-1].cpu().numpy()
        return clients_pca

    def _calculate_exec(self):
        clients_exec = {}
        exec_times = ExecutionTimeModel().predict(self.cluster.clients)
        for exec_time, client in zip(exec_times, self.cluster.clients):
            clients_exec[client.cid] = exec_time
        return clients_exec

    def map_selection(self, selection):
        selected_clients = []
        for index, client in enumerate(self.cluster.clients):
            if selection[index]:
                selected_clients.append(client)
        return selected_clients

    def map_selection2(self, selection):
        selected_clients = []
        selected_ids = [c.cid for c in selection]
        for client in self.cluster.clients:
            if client.cid in selected_ids:
                selected_clients.append(client)
        return selected_clients

    def execution_time(self, solution):
        exec_times = []
        for client in solution:
            exec_times.append(self.clients_exc[client.cid])
        return max(exec_times)[0]

    def measure_quality(self, solution):
        global_model = self.clients_pca['global']
        clients_model = []
        for client in solution:
            clients_model.append(self.clients_pca[client.cid])
        return self._distance(clients_model, global_model)

    def _distance(self, many, one):
        all_results = []
        for single in many:
            result = wasserstein_distance(one, single)
            all_results.append(result)
        results = sum(all_results) / len(all_results)
        results = abs(math.log10(results)) if results != 0 else results
        return results

    def measure_participation(self, solution):
        return len(solution)

    def evaluate(self, solution: typing.List):
        obj1 = self.execution_time(solution)
        obj2 = self.measure_quality(solution)
        obj3 = self.measure_participation(solution)
        evaluation = obj1 / (obj3 * .8 + obj2 * .2)
        return evaluation
