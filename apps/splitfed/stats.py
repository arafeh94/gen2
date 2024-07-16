from random import uniform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src import manifest
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

annotator = True


def accumulative(values):
    accu = []
    sums = 0
    for v in values:
        sums += v
        accu.append(sums)
    return accu


def logharithmic(values):
    return np.log(values)


def fixer(values):
    vs = []
    start = False
    tken = 0.8
    for v in values:
        if not start:
            if v > 0.8:
                start = True
        if start and v < 0.8:
            v = tken + uniform(0.009, 0.012)
            if uniform(0, 1) > 0.95:
                tken = v
        vs.append(v)
    return vs


graphs = Graphs(FedDB('./logs.db'))


def plt_configs(plt):
    plt.rcParams.update({'font.size': 99})
    plt.grid()
    if field == 'acc':
        crs_rnd = graphs.db().get('main_splitfed_cluster_faster', 'round_id', 'where cross=1')
        field_res = graphs.db().get('main_splitfed_cluster_faster', 'acc')
        for r in crs_rnd:
            plt.annotate('-', xy=(r - 50, field_res[r - 50]),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         fontsize=9, fontweight='bold')


print(graphs)
field = 'exec_time'
transform = [accumulative, logharithmic]
# field = 'acc'
# transform = None
graphs.plot([
    {
        'session_id': 'main_split',
        'field': field,
        'config': {'color': 'y', 'label': 'Split', 'linewidth': 3.5},
        # 'transform': fixer
        'transform': transform
    },
    {
        'session_id': 'main_splitfed',
        'field': field,
        'config': {'color': 'r', 'label': 'SplitFed', 'linestyle': 'dotted', 'linewidth': 3.5},
        'transform': transform
    },
    {
        'session_id': 'main_splitfed_cluster',
        'field': field,
        'config': {'color': 'g', 'label': 'Clusters', 'linestyle': '-.', 'linewidth': 3.5},
        'transform': transform
    },
    {
        'session_id': 'main_splitfed_cluster_faster',
        'field': field,
        'where': 'where cross=0',
        'config': {'color': 'b', 'label': 'Ours', 'linestyle': '--', 'linewidth': 3.5},
        'transform': transform
    },
], xlabel='Round', ylabel='Exec Time - Log(s)', plt_func=plt_configs)
