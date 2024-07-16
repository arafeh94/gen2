import json

import matplotlib.pyplot as plt

from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graphs = Graphs(FedDB('res.db'))
plt.rcParams.update({'font.size': 28})


def all_true(arr, item):
    for a in arr:
        if a not in item:
            return False
    return True


def sett(res, key, val):
    if key in res:
        res[key] = val


def dell(res, key):
    if key in res:
        del res[key]


def preprocess(res):
    dell(res, 'e84adbe186a30aeb316f939077ce9fd9')
    dell(res, 't64fc99a0a410a8cd889f6d1b1ee049fa')
    dell(res, 't17c392e390179003034102549819bdda')
    dell(res, 't15e5c7ae8c8a8e8edd827e264a666d8d')

    # mnist dir
    sett(res, 'd1f5ed4e5813eb58ac2fe879350364ae', 'α=10, E=1')
    return res


def plot_builder(field, *query):
    colors = ['b', 'r', '#117733', '#DDCC77']
    linestyles = ['-.', '-', 'dashdot', 'solid']
    index = 0
    res = {}
    plts = []
    for key, val in graphs.db().tables().items():
        if all_true(query, val):
            res[key] = val
    res = preprocess(res)
    print(json.dumps(res, indent=4))
    for k, v in res.items():
        plts.append({
            'session_id': k,
            'field': field,
            'config': {'color': colors[index % len(colors)], 'label': v,
                       'linestyle': linestyles[index % len(linestyles)],
                       'linewidth': 5},
            'transform': utils.smooth
        })
        index += 1
    return plts


def plt_config(plt):
    plt.grid()
    plt.xlabel('Round')
    plt.ylabel('EMD')
    plt.legend(loc='best')


root = './plts/'
# mnist
graphs.plot(plot_builder('wd', 'tt2_wd'), save_path=f'{root}wd.png',
            plt_func=plt_config,
            show=True)
# graphs.plot(plot_builder('acc', 'e25', 'lr0.1'), save_path=f'{root}acc.png',
#             plt_func=plt_config,
#             show=False)

# graphs.plot([
#     {
#         'session_id': 't34ee08a9bad0172a9122c9a1f7acc634',
#         'field': 'acc',
#         'config': {'color': 'r', 'label': 'Non-IID', 'linestyle': 'dotted'},
#     },
# ], xlabel='Round', ylabel='Accuracy')
