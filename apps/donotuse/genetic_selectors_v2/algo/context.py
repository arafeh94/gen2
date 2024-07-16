import copy
import statistics

import logging

from apps.donotuse.genetic_selectors.algo.cluster_selector import ClusterSelector
from src.apis import utils, math, federated_tools
from src.apis.extensions import Serializable, Dict
from src.data.data_container import DataContainer


class SelectorContext:
    def __init__(self, model_stats, sample_dict):
        self.model_stats = model_stats
        self.sample_dict = sample_dict
        self.solutions = Dict()
        self.clustered_models = ClusterSelector(utils.cluster(model_stats, compress_weights=False))

    def ecl(self, client_idx):
        aggregated = federated_tools.aggregate(utils.dict_select(client_idx, self.model_stats),
                                               utils.dict_select(client_idx, self.sample_dict))
        influences = []
        for key in client_idx:
            influence = math.influence_ecl(aggregated, self.model_stats[key])
            influences.append(influence)
        fitness = statistics.variance(math.normalize(influences))
        fitness = fitness * 10 ** 5
        return fitness


class InitiatorContext(Serializable):
    def __init__(self, clients_data: {int: DataContainer}, create_model: callable, saved_model_path='./saved_models'):
        super().__init__(saved_model_path)
        self.clients_data: {int: DataContainer} = clients_data
        self.model_stats = {}
        self.models = {}
        self.sample_dict = {}
        self.create_model = create_model
        self.init_model = self.create_model()
        self.logging = logging.getLogger('context')
        self.load()
        self.times = []

    def train(self, data_ratio=0, epochs=100, batch=50, lr=0.1):
        if len(self.models) > 0:
            self.logging.info("Models Loaded")
            return

        self.logging.info("Building Models --Started")

        for client_idx, data in self.clients_data.items():
            if 0 < data_ratio < 1:
                shuffled = data.shuffle().as_tensor()
                new_x = shuffled.x[0:int(len(data.x) * data_ratio)]
                new_y = shuffled.y[0:int(len(data.y) * data_ratio)]
                data = DataContainer(new_x, new_y)
            self.logging.info(f"Building Models --ClientID{client_idx}")
            model = copy.deepcopy(self.init_model)
            trained = federated_tools.train(model, data.batch(batch), epochs=epochs, lr=lr)
            print(trained)
            self.model_stats[client_idx] = trained
            self.models[client_idx] = model
            self.sample_dict[client_idx] = len(data)
        self.logging.info("Building Models --Finished")
        self.save()

    def aggregate_clients(self, client_idx):
        global_model_stats = federated_tools.aggregate(utils.dict_select(client_idx, self.model_stats),
                                                       utils.dict_select(client_idx, self.sample_dict))
        global_model = self.create_model()
        federated_tools.load(global_model, global_model_stats)
        return global_model

    def test_selection_accuracy(self, client_idx, test_data: DataContainer, title='test accuracy', output=True):
        self.logging.info('-----------------' + title + '-----------------')
        global_model = self.aggregate_clients(client_idx)
        acc_loss = federated_tools.infer(global_model, test_data.batch(8))
        if output:
            self.logging.info(f"test case:{client_idx}")
            self.logging.info(f"global model accuracy: {acc_loss[0]}, loss: {acc_loss[1]}")
        return acc_loss

    def ecl(self, client_idx):
        aggregated = federated_tools.aggregate(utils.dict_select(client_idx, self.model_stats),
                                               utils.dict_select(client_idx, self.sample_dict))
        influences = []
        for key in client_idx:
            influence = math.influence_ecl(aggregated, self.model_stats[key])
            influences.append(influence)
        fitness = statistics.variance(math.normalize(influences))
        fitness = fitness * 10 ** 5
        return fitness

    def fitness(self, client_idx):
        return self.ecl(client_idx)
