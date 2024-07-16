import typing
from typing import List

from src.federated.components.aggregators import AVGAggregator
from src.federated.federated import FederatedLearning
from src.federated.protocols import ClientSelector


class SequentialSelector(ClientSelector):
    def __init__(self, sequential_selection: List, future_selection_method: typing.Callable):
        self.sequential_selection = sequential_selection
        self.future_selection_method = future_selection_method

    def select(self, client_ids: List[int], context: FederatedLearning.Context) -> List[int]:
        if context.round_id < len(self.sequential_selection):
            return [self.sequential_selection[context.round_id]]
        return self.future_selection_method(client_ids, context)

    @staticmethod
    def continuous(cl_length, rounds):
        return list(range(cl_length)) * int(rounds / cl_length)


class DilutionAggregator(AVGAggregator):
    def aggregate(self, trainers_models_weight_dict: Dict[int, nn.ModuleDict], sample_size: Dict[int, int],
                  round_id: int) -> nn.ModuleDict:
        agg_weights = super().aggregate(trainers_models_weight_dict, sample_size, round_id)
        return agg_weights
