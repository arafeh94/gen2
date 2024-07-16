# mpiexec -n 4 python main_mpi.py
import logging
import sys
from os.path import dirname

from libs.model.collection import CNNCifar
from src.apis import federated_tools
from src.data.data_container import DataContainer

sys.path.append(dirname(__file__) + '../')

from src.data import data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
clients_data = data_loader.cifar10_10shards_100c_400min_400max().map(lambda cid, dt: dt.reshape((-1, 3, 32, 32)))

model = CNNCifar(10)
federated_tools.train(model, clients_data[0].batch(10))
print(clients_data)
