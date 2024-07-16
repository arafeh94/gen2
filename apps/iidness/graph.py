from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graphs = Graphs(FedDB('results.db'))

graphs.plot([
    {
        'session_id': 't58349_87030_label_100c_10l_600mn_600mx',
        'field': 'loss',
        'config': {'color': 'b', 'label': "{'iidness': 6.969058559918826}"},
    },
    {
        'session_id': 't67746_32382_label_100c_10l_600mn_600mx',
        'field': 'loss',
        'config': {'color': 'r', 'label': "{'iidness': 7.0495233045979955}"},
    },
    {
        'session_id': 't11211_94329_label_100c_10l_600mn_600mx',
        'field': 'loss',
        'config': {'color': 'g', 'label': "{'iidness': 6.353239890456613}"},
    },
    {
        'session_id': 't33748_59449_label_100c_10l_600mn_600mx',
        'field': 'loss',
        'config': {'color': 'yellow', 'label': "{'iidness': 2.948609475064131}"},
    },
    {
        'session_id': 't11374_36312_label_100c_10l_600mn_600mx',
        'field': 'loss',
        'config': {'color': 'cyan', 'label': "{'iidness': 3.6558652085402756}"},
    },
    {
        'session_id': 't64910_22905_label_100c_10l_600mn_600mx',
        'field': 'loss',
        'config': {'color': 'black', 'label': "{'iidness': 1.756841220606432}"},
    },
], xlabel='Round', ylabel='Accuracy')
