import typing

from apps.genetics.src import alg_genetic
from apps.genetics.src.context import SelectorContext
from src.apis.extensions import Dict
from src.federated.protocols import ModelBasedClientSelector


# noinspection PyUnresolvedReferences
class GeneticClientSelector(ModelBasedClientSelector):

    def __init__(self, initiator_context: 'InitiatorContext', nb_client, configs):
        super().__init__()
        self.initiator_context = initiator_context
        self.nb_client = nb_client
        self.configs = configs
        super(GeneticClientSelector, self)._update(initiator_context.model_stats, initiator_context.sample_dict)

    def model_based_select(self, client_ids, clients_weights: Dict[int, any], sample_sizes: Dict[int, int],
                           context: 'FederatedLearning.Context'):
        if context.round_id == 0:
            clients_weights = self.initiator_context.model_stats
            sample_sizes = self.initiator_context.sample_dict
        return self.genetic_selector(clients_weights, sample_sizes)

    def genetic_selector(self, weight_dict, sample_dict) -> typing.Callable:
        self.configs['c_size'] = self.nb_client
        context = SelectorContext(weight_dict, sample_dict)
        best, all_solutions = alg_genetic.ga(fitness=context.ecl, genes=context.clustered_models,
                                             desired=self.configs['desired_fitness'],
                                             max_iter=5, r_cross=self.configs['r_cross'],
                                             r_mut=self.configs['r_mut'], c_size=self.configs['c_size'],
                                             p_size=self.configs['p_size'])
        return best
