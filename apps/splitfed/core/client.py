import copy
import logging
import random
import time

import numpy as np
import torch
from torch.optim import Adam

from src.data.data_container import DataContainer
from src.federated.components.trainers import TorchTrainer


class Trainer(TorchTrainer):
    def __init__(self, initial_model, speed=1, lr=0.0001):
        super().__init__()
        self.device = torch.device('cuda')
        self.model = initial_model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.model_output = None
        self.is_trained = False
        self.speed = speed
        self.id = random.randint(0, 999999999)
        self.lr = lr

    def split_train(self, data: DataContainer):
        inputs, labels = (data.x, data.y) if isinstance(data, DataContainer) else (data[0], data[1])
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        self.model_output = self.model(inputs)
        out = self.model_output.clone().detach().requires_grad_(True)
        self.is_trained = True
        return out, labels

    def update_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def model_copy(self):
        return copy.deepcopy(self.model)

    def backward(self, grad):
        self.model_output.backward(grad)
        self.optimizer.step()
        self.is_trained = False


class Client(Trainer):
    def __init__(self, data: DataContainer, model, speed=1, lr=0.0001, cid=None, **configs):
        super().__init__(model, speed, lr)
        if configs is None:
            configs = {}
        self.data = data
        self.cid = cid
        self.speed = speed
        self.configs = configs
        self.exec_history = []
        self.used_keys = ['cpu_a', 'ram_a', 'disk_a', 'latency']

    def local_train(self):
        return super(Client, self).split_train(self.data)

    def set_available_resources(self, **available_resources):
        self.configs.update(available_resources)

    def randomize_resources(self, upto=0.8, rase_exp=True):
        if all(key in self.configs for key in ['cpu_c', 'ram_c', 'disk_c']):
            cpu_a, ram_a, disk_a, latency = resource_generator.generate(1, self.speed * upto)
            cpu_c, ram_c, disk_c = self.configs['cpu_c'], self.configs['ram_c'], self.configs['disk_c']
            self.set_available_resources(
                cpu_a=min(cpu_a, cpu_c),
                ram_a=min(ram_a, ram_c),
                disk_a=min(disk_a, disk_c),
                latency=latency
            )
            return True
        else:
            if rase_exp:
                raise Exception('resources capacity parameters are not defined')
            return False

    def exec_time(self, raise_exception=False):
        if all(key in self.configs for key in self.used_keys):
            cpu_coefficient = 0.1
            ram_coefficient = 0.05
            disk_coefficient = 0.01
            latency_coefficient = 0.001
            constant = 3
            execution_time = (
                    1 / (cpu_coefficient * self.configs['cpu_a']) +
                    1 / (ram_coefficient * self.configs['ram_a']) +
                    1 / (disk_coefficient * self.configs['disk_a']) +
                    # (self.configs['latency'] * latency_coefficient) +
                    constant
            )
            self._append_time(execution_time)
            return execution_time
        else:
            if raise_exception:
                raise Exception('available resources parameters are not defined')
            return 0

    def _append_time(self, exec_time):
        extracted = {key: self.configs[key] for key in self.used_keys if key in self.configs}
        extracted['exec_time'] = exec_time
        self.exec_history.append(extracted)

    def __repr__(self):
        return "speed: {}, data: {}".format(self.speed, self.data)

    @staticmethod
    def generate(data, client_model, speed=1, lr=0.0001, cid=None):
        cid = cid if cid is not None else random.randint(0, 999999999)
        cpu_c, ram_c, disk_c, latency = resource_generator.generate(1, speed)
        client = Client(data, client_model, speed, lr, cid, cpu_c=round(cpu_c[0]), ram_c=round(ram_c[0]),
                        disk_c=round(disk_c[0]), latency=latency)
        return client


# noinspection PyTypeChecker
class ResourceGenerator:
    def __init__(self):
        self.ram_mean, self.ram_std = 8, 2
        self.cpu_mean, self.cpu_std = 2, 1
        self.disk_mean, self.disk_std = 500, 100
        self.latency_mean, self.latency_std = 1000, 500

        # Add a row and column for latency in the correlation matrix
        self.correlation_matrix = np.array([[1.0, 0.7, 0.3, 0.2],
                                            [0.7, 1.0, 0.5, 0.4],
                                            [0.3, 0.5, 1.0, 0.1],
                                            [0.2, 0.4, 0.1, 1.0]])

    def generate(self, num_clients, speed):
        resources = np.random.multivariate_normal([self.ram_mean, self.cpu_mean, self.disk_mean, self.latency_mean],
                                                  [
                                                      [self.ram_std ** 2, 0, 0, 0],
                                                      [0, self.cpu_std ** 2, 0, 0],
                                                      [0, 0, self.disk_std ** 2, 0],
                                                      [0, 0, 0, self.latency_std ** 2]
                                                  ],
                                                  size=num_clients)

        # Apply the correlation matrix
        correlated_resources = np.dot(resources, np.linalg.cholesky(self.correlation_matrix).T)

        # Ensure non-negative values for resources
        correlated_resources = np.maximum(correlated_resources, [1, 0.5, 100, 1])

        scaled_resources = correlated_resources * speed

        return scaled_resources[:, 0], scaled_resources[:, 1], scaled_resources[:, 2], scaled_resources[:, 3]


resource_generator = ResourceGenerator()
