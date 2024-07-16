import numpy as np
from torch import nn
from apps.donotuse.genetic_selectors.algo.context import Context
from src.apis import utils, federated_tools


def rl_module_creator(clients_data, initial_model) -> nn.Module:
    w_first = utils.flatten_weights(initial_model.state_dict())

    context = Context(clients_data, lambda: initial_model)
    context.train(data_ratio=0.1, epochs=5)
    weights = {}
    for trainer_id, stats in context.model_stats.items():
        weights[trainer_id] = utils.flatten_weights(stats)

    aggregated = context.aggregate_clients(clients_data.keys()).state_dict()

    client_director = {}
    for trainer_id, weight in weights.items():
        delta = weight - w_first
        normalized = delta / np.sqrt(np.dot(delta, delta))
        client_director[trainer_id] = normalized

    federated_tools.load(initial_model, aggregated)
    return lambda: initial_model, client_director, w_first
