import logging
import typing
from apps.donotuse.genetic_selectors.algo.cluster_selector import ClusterSelector
from apps.genetics.src import genetic
from apps.genetics.src.context import InitiatorContext
from src.apis import utils


def ga_module_creator(clients_data, init_model, ga_configs, epoch=100, batch=50,
                      saved_models='./saved_models', lr=0.1) -> typing.Callable:
    context = InitiatorContext(clients_data, init_model, saved_models)
    context.train(1, epoch, batch, lr)
    clustered_models = utils.cluster(context.model_stats, ga_configs['nb_clusters'], False)
    clustered = ClusterSelector(clustered_models)
    best, all_solutions = genetic.ga(fitness=context.fitness, genes=clustered, desired=ga_configs['desired_fitness'],
                                     max_iter=ga_configs['max_iter'], r_cross=ga_configs['r_cross'],
                                     r_mut=ga_configs['r_mut'], c_size=ga_configs['c_size'],
                                     p_size=ga_configs['p_size'])
    logging.getLogger('ga').info(best)
    global_model = context.aggregate_clients(best)
    return context, lambda: global_model


def cluster_module_creator(clients_data, init_model, clusters=10, c_size=1):
    context = InitiatorContext(clients_data, init_model)
    context.train(data_ratio=0.1)
    clustered = ClusterSelector(utils.cluster(context.model_stats))
    selected_idx = []
    while len(selected_idx) < c_size:
        available = clustered.list()
        selected_idx.append(clustered.select(available[0]))
    global_model = context.aggregate_clients(selected_idx)
    return lambda: global_model
