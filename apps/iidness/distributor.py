import math
import random
from collections import defaultdict

import numpy as np
from numpy.random import multinomial

from src.apis.extensions import Dict
from src.data.data_container import DataContainer
from src.data.data_distributor import Distributor

STATICSSSS = 1


def rand_floats(Size):
    Scalar = 1.0
    VectorSize = Size
    RandomVector = [random.random() for i in range(VectorSize)]
    RandomVectorSum = sum(RandomVector)
    RandomVector = [Scalar * i / RandomVectorSum for i in RandomVector]
    return RandomVector


def rand_int_vec(list_size, list_sum_value, distribution='Normal', sigma=0):
    """
    Inputs:
    ListSize = the size of the list to return
    ListSumValue = The sum of list values
    Distribution = can be 'uniform' for uniform distribution, 'normal' for a normal distribution ~ N(0,1) with +/- 5 sigma  (default), or a list of size 'ListSize' or 'ListSize - 1' for an empirical (arbitrary) distribution. Probabilities of each of the p different outcomes. These should sum to 1 (however, the last element is always assumed to account for the remaining probability, as long as sum(pvals[:-1]) <= 1).
    Output:
    A list of random integers of length 'ListSize' whose sum is 'ListSumValue'.
    """
    if type(distribution) == list:
        DistributionSize = len(distribution)
        if list_size == DistributionSize or (list_size - 1) == DistributionSize:
            Values = multinomial(list_sum_value, distribution, size=1)
            OutputValue = Values[0]
    elif distribution.lower() == 'uniform':  # I do not recommend this!!!! I see that it is not as random (at least on my computer) as I had hoped
        UniformDistro = [1 / list_size for i in range(list_size)]
        Values = multinomial(list_sum_value, UniformDistro, size=1)
        OutputValue = Values[0]
    elif distribution.lower() == 'normal':
        """
        Normal Distribution Construction....It's very flexible and hideous
        Assume a +-3 sigma range.  Warning, this may or may not be a suitable range for your implementation!
        If one wishes to explore a different range, then changes the LowSigma and HighSigma values
        """
        LowSigma = -sigma  # -3 sigma
        HighSigma = sigma  # +3 sigma
        StepSize = 1 / (float(list_size) - 1)
        ZValues = [(LowSigma * (1 - i * StepSize) + (i * StepSize) * HighSigma) for i in range(int(list_size))]
        # Construction parameters for N(Mean,Variance) - Default is N(0,1)
        Mean = 0
        Var = 1
        # NormalDistro= [self.NormalDistributionFunction(Mean, Var, x) for x in ZValues]
        NormalDistro = list()
        for i in range(len(ZValues)):
            if i == 0:
                ERFCVAL = 0.5 * math.erfc(-ZValues[i] / math.sqrt(2))
                NormalDistro.append(ERFCVAL)
            elif i == len(ZValues) - 1:
                ERFCVAL = NormalDistro[0]
                NormalDistro.append(ERFCVAL)
            else:
                ERFCVAL1 = 0.5 * math.erfc(-ZValues[i] / math.sqrt(2))
                ERFCVAL2 = 0.5 * math.erfc(-ZValues[i - 1] / math.sqrt(2))
                ERFCVAL = ERFCVAL1 - ERFCVAL2
                NormalDistro.append(ERFCVAL)
                # print "Normal Distribution sum = %f"%sum(NormalDistro)
            Values = multinomial(list_sum_value, NormalDistro, size=1)
            OutputValue = Values[0]
        return OutputValue
    else:
        raise ValueError('Cannot create desired vector')
    return OutputValue


class IidDistributor(Distributor):

    def __init__(self, num_clients, label_per_client, min_size, max_size,
                 is_random_label_count=False, is_random_label_size=False):
        super().__init__()
        self.num_clients = num_clients
        self.label_per_client = label_per_client
        self.min_size = min_size
        self.max_size = max_size
        self.is_random_label_count = is_random_label_count
        self.is_random_label_size = is_random_label_size

    def distribute(self, data: DataContainer) -> Dict[int, DataContainer]:
        global STATICSSSS
        data = data.as_numpy()
        self.log(f'distributing {data}', level=0)
        clients_data = defaultdict(list)
        grouper = self.Grouper(data.x, data.y)
        sigmas = [1, 2, 3, 10, 7, 15, 20]
        sigma = sigmas[STATICSSSS % len(sigmas) - 1]
        STATICSSSS += 1
        print(sigma)
        for client_id in range(self.num_clients):
            client_data_size = random.randint(self.min_size, self.max_size)
            label_per_client = random.randint(1, self.label_per_client) if self.is_random_label_count \
                else self.label_per_client
            selected_labels = grouper.groups(label_per_client)
            if self.is_random_label_size:
                selected_data_size = rand_int_vec(len(selected_labels), client_data_size, 'normal', sigma)
            else:
                selected_data_size = [int(client_data_size / len(selected_labels))] * len(selected_labels)
            # self.log(f'generating data for {client_id}-{selected_labels}')
            client_x = []
            client_y = []
            for index, label in enumerate(selected_labels):
                data_size = selected_data_size[index]
                rx, ry = grouper.get(label, data_size)
                if len(rx) == 0:
                    self.log(f'shard {round(label)} have no more available data to distribute, skipping...', level=0)
                else:
                    client_x = rx if len(client_x) == 0 else np.concatenate((client_x, rx))
                    client_y = ry if len(client_y) == 0 else np.concatenate((client_y, ry))
            grouper.clean()
            clients_data[client_id] = DataContainer(client_x, client_y).as_tensor()
        return Dict(clients_data)

    class Grouper:
        def __init__(self, x, y):
            self.grouped = defaultdict(list)
            self.selected = defaultdict(lambda: 0)
            self.label_cursor = 0
            for label, data in zip(y, x):
                self.grouped[label].append(data)
            self.all_labels = list(self.grouped.keys())

        def groups(self, count=None):
            if count is None:
                return self.all_labels
            selected_labels = []
            for i in range(count):
                selected_labels.append(self.next())
            return selected_labels

        def next(self):
            if len(self.all_labels) == 0:
                raise Exception('no more data available to distribute')

            temp = 0 if self.label_cursor >= len(self.all_labels) else self.label_cursor
            self.label_cursor = (self.label_cursor + 1) % len(self.all_labels)
            return self.all_labels[temp]

        def clean(self):
            for label, records in self.grouped.items():
                if label in self.selected and self.selected[label] >= len(records):
                    print('cleaning the good way')
                    del self.all_labels[self.all_labels.index(label)]

        def get(self, label, size=0):
            if size is None:
                size = len(self.grouped[label])
            random_start = random.randint(0, len(self.grouped[label]) - size - 1)
            x = self.grouped[label][random_start:random_start + size]
            # x = self.grouped[label][self.selected[label]:self.selected[label] + size]
            y = [label] * len(x)
            # self.selected[label] += size
            # if len(x) == 0 and label in self.all_labels:
            #     print('cleaning the wrong way')
            #     del self.all_labels[self.all_labels.index(label)]
            return x, y

    def id(self):
        r = '_r' if self.is_random_label_count else ''
        return f'label_{self.num_clients}c_{self.label_per_client}l_{self.min_size}mn_{self.max_size}mx' + r
