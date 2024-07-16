import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src import manifest
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

rounds_pca = FedDB('cache/perf.db').get('noniid', 'pca')
# rounds_pca = FedDB('cache/perf.db').get('session92922', 'pca')
rounds_pca = [json.loads(p) for p in rounds_pca]
for i, trainers_pca in enumerate(rounds_pca):
    plt.clf()
    for t_id, pca in trainers_pca.items():
        plt.plot(pca)
    plt.pause(1 if i == 0 else 0.05)

plt.show()
