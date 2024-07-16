import typing

from apps.donotuse.genetic_selectors_v2.algo.context import SelectorContext, InitiatorContext
from apps.donotuse.genetic_selectors.algo import genetic
from src.apis.extensions import Dict
from src.federated.federated import FederatedLearning
from src.federated.protocols import ModelBasedClientSelector


class GeneticSelector(ModelBasedClientSelector):

    def __init__(self, initiator_context: 'InitiatorContext', nb_client, genetic_config):
        super().__init__()
        self.initiator_context = initiator_context
        self.nb_client = nb_client
        self.genetic_config = genetic_config
        super(GeneticSelector, self)._update(initiator_context.model_stats, initiator_context.sample_dict)

    def model_based_select(self, client_ids, clients_weights: Dict[int, any], sample_sizes: Dict[int, int],
                           context: 'FederatedLearning.Context'):
        self.genetic_config['c_size'] = self.nb_client
        if context.round_id == 0:
            clients_weights = self.initiator_context.model_stats
            sample_sizes = self.initiator_context.sample_dict
        return genetic_selector(clients_weights, sample_sizes, self.genetic_config)


def genetic_selector(weight_dict, sample_dict, config) -> typing.Callable:
    context = SelectorContext(weight_dict, sample_dict)
    best, all_solutions = genetic.ga(fitness=ajaj_fitness, genes=context.clustered_models,
                                     desired=config['desired_fitness'],
                                     max_iter=5, r_cross=config['r_cross'],
                                     r_mut=config['r_mut'], c_size=config['c_size'], p_size=config['p_size'])
    return best


def ajaj_fitness(client_idx):
    return 0

# def cluster_selector(clients_data, init_model, clusters=10, c_size=1):
#     context = Context(clients_data, init_model)
#     context.train(data_ratio=0.1)
#     clustered = ClusterSelector(context.cluster(clusters))
#     selected_idx = []
#     while len(selected_idx) < c_size:
#         available = clustered.list()
#         selected_idx.append(clustered.select(available[0]))
#     global_model = context.aggregate_clients(selected_idx)
#     return lambda: global_model
