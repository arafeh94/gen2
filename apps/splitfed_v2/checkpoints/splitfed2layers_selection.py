import copy
import json
import logging
import random
import sys
from collections import defaultdict

from apps.splitfed_v2.core import splitlearn
from apps.splitfed_v2.core.server import Server
from apps.splitfed_v2.core.splitlearn import ClsIterator, cluster
from src.apis import utils
from src.apis.extensions import Dict
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from apps.donotuse.split_learning import models
from src.federated.subscribers.sqlite_logger import SQLiteLogger

utils.enable_logging(level=logging.ERROR)
random.seed(42)
configs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {'name': '2layers_selection'}
logger = SQLiteLogger.new_instance('splitlearn_v2.sqlite', configs)
printer = logging.getLogger('2LO')
# configs
rounds = 4
client_model = models.MnistClient(784, 32, 10)
server_model = models.MnistServer(784, 32, 10)
mnist = preload('mnist', ShardDistributor(300, 2))
test_data = preload('mnist10k').as_tensor()
cluster_speeds = [.1, .25, 1]
out_clusters = 5

double_clustered = cluster(mnist, out_clusters, cluster_speeds, client_model)
servers = Dict()
history = defaultdict(dict)
for cluster_speed in cluster_speeds:
    servers[cluster_speed] = Server(copy.deepcopy(server_model), copy.deepcopy(client_model), test_data)
internal_counter = 0
for round_id in range(rounds):
    itero = ClsIterator(cluster_speeds)
    stats = None
    while True:
        cross = itero.next()
        if isinstance(cross, tuple) and all(cross[:2]):
            printer.error("aggregation: {}".format(cross))
            server_weights, clients_weights = splitlearn.crossgregate(
                servers[cross[0]].models(),
                servers[cross[1]].models(), 3)
            for speed_idx, server in servers.select([cross[0], cross[1]]).items():
                server.server_model.load_state_dict(server_weights)
                server.client_model.load_state_dict(clients_weights)
        if not cross[2]:
            break
        if len(cross) > 3 and cross[3]:
            continue
        speed_idx = itero.val()
        run_clusters = []
        for rs_key, outer in double_clustered.items():
            run_clusters.append(outer[speed_idx])
        stats = splitlearn.one_round_resource(run_clusters, servers[speed_idx], is_selection=True)
        stats['speed'] = speed_idx
        stats['round_num'] = round_id
        stats['iter_num'] = itero.counter()
        logger.log_all(internal_counter, stats)
        history[round_id][itero.counter()] = stats
        internal_counter += 1
        printer.error("speed_idx - {}: {}".format(speed_idx, stats))
        itero.append_time(stats['round_time'])
    printer.error('round_end: {}'.format(round_id))
    printer.error('------------------------------------')
