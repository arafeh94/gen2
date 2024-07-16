import logging
import typing

from apps.donotuse.genetic_selectors.algo.cluster_selector import ClusterSelector
from apps.donotuse.genetic_selectors.algo import genetic
from apps.donotuse.genetic_selectors.algo.context import Context


def ga_module_creator(clients_data, init_model, max_iter=20, r_cross=0.1, r_mut=0.05, c_size=10,
                      p_size=20, clusters=10, desired_fitness=0.5, epoch=100, batch=50,
                      saved_models='./saved_models', lr=0.1) -> typing.Callable:
    context = Context(clients_data, init_model, saved_models)
    context.train(1, epoch, batch, lr)
    clustered = ClusterSelector(context.cluster(clusters, compress=False))
    best, all_solutions = genetic.ga(fitness=context.fitness, genes=clustered, desired=desired_fitness,
                                     max_iter=max_iter, r_cross=r_cross, r_mut=r_mut, c_size=c_size, p_size=p_size)
    logging.getLogger('ga').info(best)
    global_model = context.aggregate_clients(best)
    return lambda: global_model


def cluster_module_creator(clients_data, init_model, clusters=10, c_size=1, epochs=100, batch=50, lr=0.1,
                           saved_model_path='./kcenter'):
    context = Context(clients_data, init_model, saved_model_path=saved_model_path)
    context.train(data_ratio=0.6, epochs=epochs, batch=batch, lr=lr)
    clustered = ClusterSelector(context.cluster(clusters, compress=False))
    selected_idx = []
    while len(selected_idx) < c_size:
        available = clustered.list()
        selected_idx.append(clustered.select(available[0]))
    global_model = context.aggregate_clients(selected_idx)
    return lambda: global_model


def ga_resource_selector(resources, max_iter=20, r_cross=0.1, r_mut=0.05, c_size=10,
                         p_size=20, desired_fitness=0.5) -> typing.Callable:
    clustered = ClusterSelector(resources)
    best, all_solutions = genetic.ga(fitness=fitness, genes=clustered, desired=desired_fitness,
                                     max_iter=max_iter, r_cross=r_cross, r_mut=r_mut, c_size=c_size, p_size=p_size)
    logging.getLogger('ga').info(best)
    return best


def fitness(chromosome):
    fit1 = fitness1(chromosome)
    fit2 = fitness2(chromosome)
    fit3 = fitness3(chromosome)
    return 0.1 * fit1 + fit2 + fit3


def fitness1(chromosome):
    rams = [gene['ram'] for gene in chromosome]
    return sum(rams)


def fitness2(chromosome):
    pass


def fitness3(chromosome):
    pass
