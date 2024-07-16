import copy
import logging
import math
import statistics
import typing
from typing import Callable
from collections import Counter

from numpy.random import default_rng
import random
import numpy as np

from src.apis.utils import duplicates

logger = logging.getLogger('ga')


class ClusterSelector:
    def __init__(self, cluster_cid_dict: dict):
        """
        @param cluster_cid_dict dictionary of cluster id, and the clients ids inside the cluster
        """
        self.cluster_cid_dict = cluster_cid_dict
        self.used_clusters = []
        self.used_ids = []

    def reset(self):
        self.used_clusters = []
        self.used_ids = []

    def select(self, cid):
        if cid in self.used_ids:
            return False
        cluster_id = None
        for cluster, cids in self.cluster_cid_dict.items():
            if cid in cids:
                cluster_id = cluster
        self.used_clusters.append(cluster_id)
        self.used_ids.append(cid)
        return cid

    def list(self, but=None):
        but = but or []
        available = []
        for cluster_id, cids in self.cluster_cid_dict.items():
            if cluster_id not in self.used_clusters:
                for cid in cids:
                    if cid not in self.used_ids and cid not in but:
                        available.append(cid)
        if len(available) == 0:
            self.reset()
            return self.list(but)
        return available

    def __len__(self):
        return sum([len(cids) for cluster, cids in self.cluster_cid_dict.items()])


def select_random(genes: ClusterSelector, size):
    if len(genes) < size:
        raise Exception("genes size is less than the requested population size")
    rng = default_rng()
    selected = []
    genes.reset()
    while len(selected) < size:
        available = genes.list(selected)
        random_choice = rng.choice(len(available), size=1, replace=False)[0]
        selected_id = genes.select(available[random_choice])
        if selected_id is not False:
            selected.append(selected_id)
    return selected


def build_population(genes: ClusterSelector, p_size, c_size: typing.Union[int, tuple]):
    population = []
    for i in range(p_size):
        chromosome_size = random.randint(c_size[0], c_size[1]) if isinstance(c_size, tuple) else c_size
        chromosome = select_random(genes, chromosome_size)
        population.append(chromosome)
    return population


def crossover(p1, p2, r_cross):
    if len(p1) < 3 or len(p2) < 3:
        return [p1, p2]
    c1, c2 = p1.copy(), p2.copy()
    if random.uniform(0, 1) < r_cross:
        pt = np.random.randint(1, len(p1) - 1)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def mutation(arr, genes, r_mut):
    copy = arr.copy()
    for index, value in enumerate(copy):
        if random.uniform(0, 1) < r_mut:
            copy[index] = select_random(genes, 1)[0]
    return copy


def selection(population, scores, ratio=0.5):
    selected = []
    selected_indexes = []
    prob = wheel(population, scores)
    while len(selected) < len(population) * ratio:
        for index, item in enumerate(population):
            if prob[index] > random.uniform(0, 1) and index not in selected_indexes:
                selected.append(item)
                selected_indexes.append(index)
                if len(selected) >= len(population) * ratio:
                    break
    return selected


def populate(population, p_size):
    copy = population.copy()
    while len(copy) < p_size:
        p1 = np.random.randint(0, len(population))
        p2 = np.random.randint(0, len(population))
        pn = crossover(population[p1], population[p2], 1)
        copy += pn
    while len(copy) > p_size:
        copy.pop()
    return copy


def wheel(population, scores):
    total = np.sum(scores)
    return [1 - (scores[index] / total) for index, item in enumerate(population)]


def wheel2(population, scores):
    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    total_fitness = sum(normalized_scores)
    selection_probabilities = [1 - (fit / total_fitness) for fit in normalized_scores]
    return selection_probabilities


def normalize(arr):
    total = math.fsum(arr)
    return [i / total for i in arr]


def has_duplicate(lst):
    return len(duplicates(lst)) > 0


def clean(population):
    temp = []
    for index, item in enumerate(population):
        if not has_duplicate(item):
            temp.append(item)
    if len(temp) % 2 != 0:
        temp.pop()
    for i in temp:
        if has_duplicate(i):
            clean(population)
    return temp


def ga(fitness, genes: ClusterSelector, desired, max_iter, r_cross=0.1, r_mut=0.05, c_size=20, p_size=10,
       post_fitness: Callable = None) -> (list, [list]):
    population = build_population(genes, p_size, c_size)
    solution = None
    all_solutions = []
    minimize = 99999999999
    n_iter = 0
    while n_iter < max_iter and minimize > desired:
        scores = [fitness(chromosome) for chromosome in population]
        scores = post_fitness(scores) if post_fitness is not None else scores
        for index, ch in enumerate(population):
            if scores[index] < minimize:
                minimize = scores[index]
                solution = ch
                all_solutions.append(ch)
                logger.info(f"Solution Found: {solution} Fitness: {minimize}")
        population = selection(population, scores, ratio=0.5)
        population = populate(population, int(p_size * 3 / 4))
        population += build_population(genes, p_size - len(population), c_size)
        children = list()
        for i in range(0, len(population), 2):
            p1, p2 = population[i], population[i + 1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, genes, r_mut)
                children.append(c)
        population = clean(children)
        n_iter += 1
    return solution, all_solutions


def variance(arr):
    return statistics.variance(normalize(arr))
