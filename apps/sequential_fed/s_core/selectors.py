import logging
import random
from abc import abstractmethod
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from apps.genetics.src import alg_genetic
from apps.sequential_fed.s_core import cluster_creator
from apps.sequential_fed.s_core.cluster_creator import ClusterCreator
from src.apis import utils
from src.apis.utils import group_dict_by_values, normalize_by_column

logger = logging.getLogger(__name__)


class SelectIterator:
    def __init__(self, train_clients: dict):
        self.train_clients = train_clients
        self.idle_buffer = []
        self.select_buffer = []
        self.selection_history = defaultdict(int)

    @abstractmethod
    def build(self, **params):
        pass

    @abstractmethod
    def reset(self):
        pass

    def next(self):
        selection = self.idle_buffer.pop()
        self.select_buffer.append(selection)
        self.selection_history[selection] += 1
        return len(self.idle_buffer) != 0

    def get_id(self) -> int:
        return self.select_buffer.pop(0)

    def top_n(self, n, printer=None):
        if printer:
            printer(f"Top {n}: {[(cid, self.train_clients[cid], val) for cid, val in self.top_n(n).items()]}")
        return dict(sorted(self.selection_history.items(), key=lambda item: item[1], reverse=True)[:n])


class SeqAllClientSelector(SelectIterator):
    def build(self):
        self.idle_buffer = list(self.train_clients.keys())

    def reset(self):
        self.build()


class SeqRandomClientSelector(SelectIterator):
    def __init__(self, train_clients: dict, size=10):
        super().__init__(train_clients)
        self.size = size

    def build(self):
        self.idle_buffer = list(self.train_clients.keys())
        self.idle_buffer = np.random.choice(self.idle_buffer, size=self.size, replace=False).tolist()

    def reset(self):
        self.build()


class SeqGAClientSelector(SelectIterator):
    def __init__(self, train_clients: dict, selection_size, nb_clusters, buffer):
        super().__init__(train_clients)
        self.clusters = {}
        self.already_selected_clients = []
        self.cluster_creator = None
        self.selection_size = selection_size
        self.nb_clusters = nb_clusters
        self.buffer = buffer

    def build(self, base_model):
        self.cluster_creator = ClusterCreator(self.train_clients, base_model)
        self.cluster_creator.train(1, 0.1)
        self.clusters = self.cluster_creator.cluster(self.nb_clusters)
        self.reset()

    def selection_times(self, client_id):
        return self.selection_history[client_id] if client_id in self.selection_history else 0

    def pop_buffer(self):
        if len(self.already_selected_clients) + self.selection_size >= len(self.train_clients):
            self.already_selected_clients.clear()
        buffer_clusters = {}
        for label, clients in self.clusters.items():
            selected = [item for item in self.clusters[label] if item not in self.already_selected_clients]
            buffer_clusters[label] = selected
        return buffer_clusters

    def proba_buffer(self, acceptance_mean=0.2):
        buffer_clusters = {}
        for label, clients in self.clusters.items():
            cluster_clients = self.clusters[label]
            selection_proba = [1 / (1 + self.selection_times(client_id)) for client_id in cluster_clients]
            selected = [item for item, proba in zip(cluster_clients, selection_proba) if
                        proba > np.random.normal(loc=acceptance_mean)]
            buffer_clusters[label] = selected
        return buffer_clusters

    def reset(self):
        buffer_clusters = self.proba_buffer() if self.buffer == 'proba' else self.pop_buffer()
        cluster_selector = alg_genetic.ClusterSelector(buffer_clusters)
        self.idle_buffer, _ = alg_genetic.ga(self.fitness, cluster_selector, -999, 50, c_size=self.selection_size,
                                             post_fitness=self.post_fitness)
        self.already_selected_clients.extend(self.idle_buffer)
        utils.shuffle(self.idle_buffer)
        logger.info(f"selected_clients: {self.idle_buffer}")

    def fitness(self, client_idx):
        sim_index = 1 / self.convert_to_01(self.measure_similarity(client_idx))
        size = 1 / np.mean([len(dt) for dt in utils.dict_select(client_idx, self.train_clients).values()])
        return sim_index, size

    def post_fitness(self, scores):
        scores_0 = [s[0] for s in scores]
        scores_1 = [s[1] for s in scores]
        scores_0 = utils.normalize_array(scores_0)
        scores_1 = utils.normalize_array(scores_1)
        results = []
        for index, score in enumerate(zip(scores_0, scores_1)):
            measured_score = score[0] * 0.2 + score[1] * 0.8
            results.append(measured_score)
        return results

    def convert_to_01(self, val):
        new_value = (val + 1) / 2
        return new_value

    def measure_similarity(self, client_idx):
        compressed_weights = utils.dict_select(client_idx, self.cluster_creator.compressed_weights)
        weights_array = np.array(list(compressed_weights.values()))
        sim_matrix = cosine_similarity(weights_array)
        sim_index = np.mean(sim_matrix)  # higher indicate more similarities
        return sim_index


def create(sid, train_clients, config):
    if sid == 'ga':
        return SeqGAClientSelector(train_clients, config.wmp.cr, config.wmp.cls, config.wmp.buffer)
    elif sid == 'rn':
        return SeqRandomClientSelector(train_clients, config.wmp.cr)
    else:
        return SeqAllClientSelector(train_clients)
