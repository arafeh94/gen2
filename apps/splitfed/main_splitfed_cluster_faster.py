import copy
import os
from apps.splitfed.core.splitfed import SplitFed
from apps.splitfed.core import clusters, dist
from apps.splitfed.models import MnistClient, MnistServer
from src.apis import lambdas
from src.data.data_loader import preload
from src.federated.subscribers.sqlite_logger import SQLiteLogger

logger = SQLiteLogger(os.path.splitext(os.path.basename(__file__))[0], 'logs.db')

# the max number of client in each cluster, use to avoid accuracy problem and cat. for.
cluster_limit = 4
# initialize clients, it should be replaced with weight divergence analysis.
# In this case, we have 5 clusters in each we have x clients
client_cluster_size = 20

# sr=slow_rate: probability of a created clients to be considered as slow with low capabilities
# increase virtual training time
rounds = 50

# init data
client_model = MnistClient(784, 32, 10)
server_model = MnistServer(784, 32, 10)
clients_data = preload('mnist', dist.mnist_clustered(client_cluster_size, 300),
                       tag=f'cluster{client_cluster_size}p{300}')
train_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[0]).map(lambdas.as_tensor)
test_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[1]).reduce(lambdas.dict2dc).as_tensor()

# split learning
fast, slow = clusters.generate_speed(train_data, client_model, client_cluster_size, cluster_limit, cllr=0.001)
split_fast = SplitFed(copy.deepcopy(server_model), copy.deepcopy(client_model), fast, test_data, 1, 0.001,
                      is_cluster=True)
split_slow = SplitFed(copy.deepcopy(server_model), copy.deepcopy(client_model), slow, test_data, 1, 0.001,
                      is_cluster=True)
split_slow.one_round()
r = 0
while rounds > 0:
    st = split_slow.one_round()
    rest = st['round_exec_time']
    logger.log(r + rounds, acc=st['acc'], exec_time=st['round_exec_time'], cross=1)
    # print('slow', 'acc:', st['acc'], 'in:', st['round_exec_time'])
    print('slow:', rest)
    while rest > 0:
        stats = split_fast.one_round()
        ex = stats['round_exec_time']
        logger.log(r, acc=stats['acc'], exec_time=stats['round_exec_time'], cross=0)
        print('acc:', stats['acc'], 'in:', stats['round_exec_time'])

        rest -= ex
        # print('fast', 'speed:', ex, 'rest:', rest)
        rounds -= 1
        r += 1
        if rounds <= 0:
            break
    # cross aggregate
    # print('async aggregation running, merging slow clients with fast server')
    # print('acc before async:', split_fast.infer())
    split_fast.crossgregate2(split_slow)
    acc = split_fast.infer()
    print('acc after crossgregate:', acc)
    logger.log(r + 50, acc=acc, exec_time=st['round_exec_time'], cross=1)

print('total exec time:', sum(split_fast.round_exec_times))
