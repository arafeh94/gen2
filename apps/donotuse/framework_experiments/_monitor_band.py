import matplotlib.pyplot as plt

from apps.donotuse.framework_experiments import tools

plt.rcParams.update({'font.size': 28})

queries = [
    ['band_mnist'],

]

tables = tools.filter(queries[0][0]).keys()
each = 100
band_log1 = []
band_log2 = []


def trans1(vals):
    for index, val in enumerate(vals):
        if not index % each or index == 10 or index == len(vals) - 1:
            print('asd')
            band_log1.append((round(424 * index * 10 / 1000_000, 3), index, val))
    return vals


def trans2(vals):
    for index, val in enumerate(vals):
        if not index % each or index == 10 or index == len(vals) - 1:
            band_log2.append((round(804 * index * 10 / 1000_000, 3), index, val))
    return vals


def plt_conf2(plt):
    tools.plt_config(plt)
    size = 12
    for b in band_log1:
        plt.annotate(str(b[0]), xy=(b[1], b[2]), fontsize=size)
    for b in band_log2:
        plt.annotate(str(b[0]), xy=(b[1], b[2]), fontsize=size)


tools.graphs.plot([
    {
        'session_id': 't1145cdb99d6409a5fbb4ed2c32753c97',
        'field': 'acc',
        'config': {'color': 'b', 'label': 'LogR'},
        'transform': trans1,
    },
    {
        'session_id': 't4324a38fc3b537e5ba54bfe7686c5447',
        'field': 'acc',
        'config': {'color': 'r', 'label': 'CNN', 'linestyle': '--'},
        'transform': trans2,
    }
], plt_func=plt_conf2)
print(tables)
