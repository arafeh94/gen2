import hashlib
import logging
import os
import random
import time
import typing
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from functools import reduce
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import wasserstein_distance
from sklearn import decomposition
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

from src.apis.extensions import Dict

logger = logging.getLogger('utils')


def pca(models: typing.Dict[int, any]) -> typing.Dict[int, any]:
    flattened_weights = [flatten_state_dict(model.state_dict()) for _, model in models.items()]

    stacked_weights = torch.stack(flattened_weights).cuda()
    pca_result = torch.pca_lowrank(stacked_weights)
    transformed = pca_result[0]
    clients_pca = {}
    for idx, client in models.items():
        clients_pca[idx] = transformed[idx].cpu().numpy()

    return clients_pca


def group_dict_by_values(client_dict):
    grouped_dict = {}
    for client_id, labels in client_dict.items():
        if isinstance(labels, list):
            label_key = tuple(labels)
        else:
            label_key = labels
        if label_key not in grouped_dict:
            grouped_dict[label_key] = [client_id]
        else:
            grouped_dict[label_key].append(client_id)
    return grouped_dict


def normalize_by_column(arr):
    try:
        arr = np.array(arr)
        min_values = arr.min(axis=0)
        max_values = arr.max(axis=0)
        for i in range(len(min_values)):
            if min_values[i] == max_values[i]:
                max_values[i] += 1
        normalized_array = (arr - min_values) / (max_values - min_values)
        return normalized_array
    except RuntimeWarning as e:
        raise Exception("Error during normalization: {}".format(e))


def normalize_array(arr):
    min_value = 0
    max_value = max(arr)
    if min_value == max_value:
        max_value += 1
    results = []
    for item in arr:
        val = (item - min_value) / (max_value - min_value)
        results.append(val)
    return results


def min_max_normalize_2d_array(matrix):
    num_rows, num_columns = len(matrix), len(matrix[0])
    normalized_matrix = []

    for col in range(num_columns):
        column_values = [row[col] for row in matrix]
        min_val, max_val = min(column_values), max(column_values)

        # Handle the case where all values in the column are the same
        if min_val == max_val:
            normalized_column = [0.5] * num_rows  # Set to a default value (e.g., 0.5)
        else:
            normalized_column = [(x - min_val) / (max_val - min_val) for x in column_values]

        normalized_matrix.append(normalized_column)

    # Transpose the normalized matrix to get it back in the original format
    normalized_matrix = list(map(list, zip(*normalized_matrix)))
    return normalized_matrix


def z_score_normalize_by_column(matrix):
    num_columns = len(matrix[0])
    normalized_matrix = []

    for col in range(num_columns):
        column_values = [row[col] for row in matrix]
        mean_val = sum(column_values) / len(column_values)
        std_dev = (sum((x - mean_val) ** 2 for x in column_values) / len(column_values)) ** 0.5
        normalized_column = [(x - mean_val) / std_dev for x in column_values]
        normalized_matrix.append(normalized_column)

    # Transpose the normalized matrix to get it back in the original format
    normalized_matrix = list(map(list, zip(*normalized_matrix)))
    return normalized_matrix


def smooth(vals, sigma=2):
    from scipy.ndimage import gaussian_filter1d
    return list(gaussian_filter1d(vals, sigma=sigma))


def hash_string(string: str):
    full_hash = str.encode(string)
    return hashlib.md5(full_hash).hexdigest()


def factors(n):
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))


# noinspection PyUnresolvedReferences
def fed_avg(runs: typing.List['FederatedLearning.Context']):
    from collections import defaultdict
    import numpy as np
    avg_acc = defaultdict(list)
    avg_loss = defaultdict(list)
    for run in runs:
        for round_id, performance in run.history.items():
            avg_acc[round_id].append(performance['acc'])
            avg_loss[round_id].append(performance['loss'])

    for round_id in avg_acc:
        avg_acc[round_id] = np.average(avg_acc[round_id])
        avg_loss[round_id] = np.average(avg_loss[round_id])
    return avg_acc, avg_loss


def validate_path(file_path):
    parent_dir = os.path.dirname(file_path)
    if len(parent_dir) and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def cluster(client_weights: Dict, cluster_size=10, compress_weights=True):
    logger.info("Clustering Models --Started")
    weights = []
    client_ids = []
    clustered = {}
    for client_id, stats in client_weights.items():
        client_ids.append(client_id)
        weights.append(compress(flatten_weights(stats), 4)
                       if compress_weights else flatten_weights(stats))
    kmeans: KMeans = KMeans(n_clusters=cluster_size).fit(weights)
    logger.info(kmeans.labels_)
    for i, label in enumerate(kmeans.labels_):
        clustered[client_ids[i]] = label
    logger.info("Clustering Models --Finished")
    return clustered


def hc_clustering(clients_weights, n_cluster, compress_weights=False):
    logger.info("Clustering Models --Started")
    weights = []
    client_ids = []
    clustered = {}
    for client_id, stats in clients_weights.items():
        client_ids.append(client_id)
        weights.append(compress(flatten_weights(stats), 4)
                       if compress_weights else flatten_weights(stats))
    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='ward')
    hc_results = hierarchical_cluster.fit(weights)
    logger.info(hc_results.labels_)
    for i, label in enumerate(hc_results.labels_):
        clustered[client_ids[i]] = label
    logger.info("Clustering Models --Finished")
    return clustered


def compress(weights, n_components):
    pca = decomposition.PCA(n_components)
    pca.fit(weights)
    weights = pca.transform(weights)
    return weights.flatten()


def compress_weights(weights, n_components):
    return compress(flatten_weights(weights), n_components)


def flatten_weights(weights):
    return np.concatenate([weights.flatten() for weights in weights.values()])


def flatten_state_dict(model_state_dict):
    flattened_params = []

    for param_tensor in model_state_dict:
        param = model_state_dict[param_tensor]
        flattened_params.append(param.view(-1))

    flattened_tensor = torch.cat(flattened_params)

    return flattened_tensor


def get_average_weight_divergence(global_weights: typing.OrderedDict,
                                  local_weights: typing.List[typing.OrderedDict]):
    all_results = []
    flattened_global_weights = flatten_weights(global_weights)
    for weight in local_weights:
        flattened_trainer_weights = flatten_weights(weight)
        result = wasserstein_distance(flattened_global_weights, flattened_trainer_weights)
        all_results.append(result)
    all_results = sum(all_results) / len(all_results)
    return all_results


def dict_select(idx, dict_ref):
    new_dict = {}
    for i in idx:
        new_dict[i] = dict_ref[i]
    return new_dict


def models_state(models):
    if isinstance(models, list):
        return [model.state_dict() for model in models]
    if isinstance(models, dict):
        return Dict(models).map(lambda _, model: model.state_dict())


def timed_func(seconds, callable: typing.Callable):
    stop = datetime.now() + timedelta(seconds=seconds)
    while datetime.now() < stop:
        callable()


def enable_logging(file_name=None, level=logging.INFO):
    if file_name:
        logging.basicConfig(filename=file_name, filemode='w', datefmt='%H:%M:%S', level=level)
    else:
        logging.basicConfig(level=level)


def swap(a, b):
    return b, a


def str_all_in(arr, str):
    if not isinstance(arr, list):
        arr = [arr]
    return all([a in str for a in arr])


def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        logger.info(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
            model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            logger.info(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            logger.info(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
    return True


class UniqueSelector:
    def __init__(self, iterable):
        self.iterable = iterable
        self.selected = []

    def select(self, index):
        if index in self.selected:
            raise Exception('inquired index have been already selected')
        self.selected.append(index)
        return self.iterable[index]

    def select_random(self):
        r = random.randint(0, len(self.iterable) - 1)
        while r in self.selected:
            r = random.randint(0, len(self.iterable) - 1)
        return self.select(r)

    def peek(self, reset=False):
        select_index = 0 if len(self.selected) == 0 else max(self.selected) + 1
        if select_index >= len(self.iterable):
            if reset:
                self.reset()
                logger.info(f"fn[peek][reset]")
                select_index = 0
            else:
                raise Exception('no more items left in the list')
        return self.select(select_index)

    def reset(self):
        self.selected = []


def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


def inverse_dict(d):
    i = defaultdict(list)
    for key, value in d.items():
        for element in value:
            i[element].append(key)
    return i


def plot_bar_dictionary(dictionary):
    labels = list(dictionary.keys())
    values = list(dictionary.values())

    plt.bar(labels, values)
    plt.xlabel('Labels')
    plt.ylabel('Values')
    plt.title('Dictionary Plot')
    plt.show()


def duplicates(lst):
    return [item for item, count in Counter(lst).items() if count > 1]


def shuffle(arr):
    return np.random.shuffle(arr)


class TimeCheckpoint:
    def __init__(self):
        self.last_checkpoint = time.time()
        self.times = []

    def checkpoint(self):
        current = time.time()
        diff = current - self.last_checkpoint
        self.last_checkpoint = current
        self.times.append(diff)
        return diff

    def cp(self, index=None):
        if index is None:
            return self.checkpoint()
        return self.times[index]
