import copy
import os
import pickle
import random
from typing import List, Dict

from apps.splitfed_v2.core.client import Client


class Cluster:

    def __init__(self, model, clients: List[Client]):
        self.clients = clients
        self.model = model

    def append(self, client: Client):
        self.clients.append(client)

    def get_trainers(self):
        trainers = []
        for client in self.clients:
            if client.is_trainable:
                trainers.append(client)
                client.train_history += 1
        # print("selected trainers are: ", [trainer.cid for trainer in trainers])
        return trainers

    def rand_resource(self):
        for client in self.clients:
            client.randomize_resources(1, False)

    def __repr__(self):
        if len(self.clients) > 0:
            return f'Cluster: {self.clients[0].__repr__()}'
        else:
            return "Empty Cluster"

    def update_model(self, new_model):
        self.model.load_state_dict(new_model.state_dict())
        for client in self.clients:
            client.model.load_state_dict(new_model.state_dict())


