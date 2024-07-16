import os

from apps.splitfed.core.splitfed import SplitFed
from apps.splitfed.core import clusters, dist
from apps.splitfed.models import MnistClient, MnistServer
from src.apis import lambdas
from src.data.data_loader import preload
from src.federated.subscribers.sqlite_logger import SQLiteLogger

# the max number of client in each cluster, use to avoid accuracy problem and cat. for.
cluster_limit = 4
# initialize clients, it should be replaced with weight divergence analysis.
# In this case, we have 5 clusters in each we have x clients
client_cluster_size = 20
rounds = 50
# init tools
logger = SQLiteLogger(os.path.splitext(os.path.basename(__file__))[0], 'logs.db')
# init data
client_model = MnistClient(784, 32, 10)
server_model = MnistServer(784, 32, 10)
clients_data = preload('mnist', dist.mnist_clustered(client_cluster_size, 300), tag=f'cluster{client_cluster_size}p{300}')
train_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[0]).map(lambdas.as_tensor)
test_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[1]).reduce(lambdas.dict2dc).as_tensor()

# split learning
fast, slow = clusters.generate_speed(train_data, client_model, 0, 0, cllr=0.001)
fast[0].merge(slow[0])
split_learning = SplitFed(server_model, client_model, fast, test_data, 1, lr=0.001, is_cluster=True)
r = 0
while rounds > 0:
    rounds -= 1
    stats = split_learning.one_round()
    print('acc:', stats['acc'], 'in:', stats['round_exec_time'])
    logger.log(r, acc=stats['acc'], exec_time=stats['round_exec_time'])
    r += 1
