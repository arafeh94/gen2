import json

from matplotlib import pyplot as plt

from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

db_name = 'dbs/res1.db'
db = FedDB(db_name)
graph = Graphs(db)

graph.plot([
    {
        'session_id': 'genetic_cluster_test_genetics',
        'field': 'acc',
        'config': {'color': 'b', 'label': 'Our Approach'},
    },
    # {
    #     'session_id': 'genetic_cluster_test_clustered',
    #     'field': 'acc',
    #     'config': {'color': 'r', 'label': 'Clustering'},
    # },
    # {
    #     'session_id': 'genetic_cluster_test_normal',
    #     'field': 'acc',
    #     'config': {'color': 'pink', 'label': 'Normal'},
    # },
], save_path=f'./{db_name}.png', plt_func=change)


