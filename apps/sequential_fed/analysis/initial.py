import json

import numpy as np
from matplotlib import pyplot as plt

from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graphs = Graphs(FedDB('../seqfed.sqlite'))
graphs.plot([
    {
        'session_id': 't1709676877',
        'field': 'acc',
        'config': {'color': 'r'}
    },
    {
        'session_id': 't1709691850',
        'field': 'acc',
        'config': {'color': 'b'}
    },
])
