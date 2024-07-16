import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src import manifest
from src.apis import utils, transformers, math
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graphs = Graphs(FedDB('./res.db'))
plt.rcParams.update({'font.size': 28})
print(graphs)


# mnist
# t5c2eed7f3d77070f46edd871605866e1 - kcenter
# t5ea3353aac013575bce36562b556b9ee - genetic
# t3be38137c8eaf64eb0b48a06cae5723f - base


# cifar
# abe73a2508f4ed29e0969942d7743afa	base-cifar10_e1_b999_r500_dis#pipe_cifar10_cr01_lr0.001
# e55d4f31ae17280900b7899dec0c4a27	kcenter-cifar10_e1_b999_r500_dis#pipe_cifar10_cr01_lr0.001
# t72efdd7948b1660323483e95daabdd3a	genetic-cifar10_e1_b999_r500_dis#pipe_cifar10_cr01_lr0.001

def plt_config(plt):
    # plt.rcParams.update({'font.size': 28})
    # plt.grid()
    plt.legend(loc='best')


freq = 1
graphs.plot([
    {
        'session_id': 't5c2eed7f3d77070f46edd871605866e1',
        'field': 'acc',
        'config': {'color': 'r', 'label': 'K-Center', 'linewidth': 2},
        'transform': [math.smooth]
    },
    {
        'session_id': 't5ea3353aac013575bce36562b556b9ee',
        'field': 'acc',
        'config': {'color': 'b', 'label': 'Genetic', 'linestyle': 'dotted', 'linewidth': 2},
        'transform': [math.smooth]
    },
    {
        'session_id': 't3be38137c8eaf64eb0b48a06cae5723f',
        'field': 'acc',
        'config': {'color': 'y', 'label': 'Base', 'linestyle': '--', 'linewidth': 2},
        'transform': [math.smooth]
    },
], xlabel='Round', ylabel='Accuracy', plt_func=plt_config)
