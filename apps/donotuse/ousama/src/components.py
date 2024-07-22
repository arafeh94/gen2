import random
from typing import List, Dict

import torch
from torch import nn

from src.app.prebuilt import FastFed
from src.federated.components.aggregators import AVGAggregator
from src.federated.protocols import ClientSelector


class Client:
    def __init__(self, id, is_malicious, reputation):
        self.id = id
        self.is_malicious = is_malicious
        self.reputation = reputation


class ContraSelector(ClientSelector):
    def __init__(self, clients: [Client], selection_size, bad_ratio):
        self.clients = clients
        self.g_clients = [c.id for c in self.clients if c.is_malicious]
        self.b_clients = [c.id for c in self.clients if not c.is_malicious]
        self.size = selection_size
        self.bad_ratio = bad_ratio

    def select(self, client_ids: List[int], context: 'FederatedLearning.Context') -> List[int]:
        b_size = self.size * self.bad_ratio
        g_size = self.size - b_size
        selected = random.sample(self.g_clients, int(g_size))
        selected.extend(random.sample(self.b_clients, int(b_size)))
        return selected


class ContraAggregator(AVGAggregator):

    def __init__(self, clients: [Client]):
        self.clients = clients

    def ppa(self, trained_models_weights: Dict[int, nn.ModuleDict]):
        # max badla bel code lli 3indak
        flattened_weights = [torch.flatten(tw) for tw in trained_models_weights]
        return {1: 1}

    def update_client_score(self, ppa_scores):
        for client in self.clients:
            ppa_score = ppa_scores[client.id]
            # update client reputation here
            client.reputation = ppa_score * 1
            pass
        pass

    def get_weight_score(self, ppa_scores: Dict[int, float]) -> Dict[int, float]:
        weight_scores = {}
        for client in self.clients:
            weight_scores[client.id] = client.reputation * ppa_scores[client.id]
        return {1: 1}

    def aggregate(self, trainers_models_weight_dict: Dict[int, nn.ModuleDict], sample_size: Dict[int, int],
                  round_id: int) -> nn.ModuleDict:
        ppa_scores = self.ppa(trainers_models_weight_dict)
        self.update_client_score(ppa_scores)
        weight_score = self.get_weight_score(ppa_scores)
        for client_id, client_weight in trainers_models_weight_dict.items():
            trainers_models_weight_dict[client_weight] *= weight_score[client_id]

        return super().aggregate(trainers_models_weight_dict, sample_size, round_id)