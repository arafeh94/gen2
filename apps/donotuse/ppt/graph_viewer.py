import sys

import matplotlib.pyplot as plt
import numpy as np
from src import manifest
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def variance(data, each=30):
    var_data = []
    results = []
    while len(var_data) < each:
        var_data.append(data[len(var_data)])
        results.append(0)
    for d in data[each:]:
        results.append(np.var(var_data))
        var_data.pop(0)
        var_data.append(d)
    return normalize(results)


graphs = Graphs(FedDB('./cache/perf.db'))
print(graphs)
graphs.plot([{
    'session_id': 'noniid',
    'field': 'acc',
    'config': {'color': 'r', 'linestyle': '--'},
},
], animated=0.05, plt_func=change)

graphs.plot([{
    'session_id': 'noniid',
    'field': 'loss',
    'config': {'color': 'r', 'linestyle': '--'},
},
], animated=0.05, plt_func=change)
graphs.plot([{
    'session_id': 'noniid',
    'field': 'wd',
    'config': {'color': 'r', 'linestyle': '--'},
},
], animated=0.05, plt_func=change)
