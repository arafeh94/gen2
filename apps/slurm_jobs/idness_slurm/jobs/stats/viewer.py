import matplotlib
from matplotlib import pyplot as plt

from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

matplotlib.rcParams.update({'font.size': 28})
plt.grid(True)

db1 = FedDB('dt.db')
graphs = Graphs(db1)
print(graphs)
field = 'loss'
field_title = 'Loss'
# mnist
graphs.plot([
    {
        'session_id': 't1648855787',
        'field': field,
        'config': {'color': 'b', 'label': '6.93', 'marker': 'o'},
    },
    {
        'session_id': 't1648865262',
        'field': field,
        'config': {'color': 'r', 'label': '6.98', 'marker': '*'},
    },
    {
        'session_id': 't1648878571',
        'field': field,
        'config': {'color': 'g', 'label': '5.87', 'marker': 'x'},
    },
    {
        'session_id': 't1648896767',
        'field': field,
        'config': {'color': 'yellow', 'label': '2.86', 'marker': '>'},
    },
    {
        'session_id': 't1648920298',
        'field': field,
        'config': {'color': 'pink', 'label': '3.71', 'marker': '<'},
    },
    {
        'session_id': 't1648946379',
        'field': field,
        'config': {'color': 'gray', 'label': '1.71', 'marker': '^'},
    },
], xlabel='Round', ylabel=field_title)
plt.grid()

# cifar
graphs.plot([
    {
        'session_id': 't1648856546',
        'field': field,
        'config': {'color': 'b', 'label': '7.14', 'marker': 'o'},
    },
    {
        'session_id': 't1648875801',
        'field': field,
        'config': {'color': 'r', 'label': '7.12', 'marker': '*'},
    },
    {
        'session_id': 't1648898008',
        'field': field,
        'config': {'color': 'g', 'label': '6.26', 'marker': 'x'},
    },
    {
        'session_id': 't1648923075',
        'field': field,
        'config': {'color': 'yellow', 'label': '2.80', 'marker': '>'},
    },
    {
        'session_id': 't1648949608',
        'field': field,
        'config': {'color': 'pink', 'label': '3.62', 'marker': '<'},
    },
    {
        'session_id': 't1648980833',
        'field': field,
        'config': {'color': 'gray', 'label': '1.73', 'marker': '^'},
    },
], xlabel='Round', ylabel=field_title)
