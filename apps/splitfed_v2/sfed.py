import copy
import json
import logging
import random
import sys
from collections import defaultdict

from apps.splitfed_v2._run_configs import global_configs
from apps.splitfed_v2.core import splitlearn
from apps.splitfed_v2.core.clusters import Cluster
from apps.splitfed_v2.core.funcs import SQLoggerCustom
from apps.splitfed_v2.core.server import Server
from apps.splitfed_v2.core.splitlearn import ClsIterator, cluster, get_clients
from src.apis import utils
from src.apis.extensions import Dict
from src.federated.components import aggregators, metrics, client_selectors
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.components.trainers import TorchTrainer
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.sqlite_logger import SQLiteLogger

utils.enable_logging(level=logging.INFO)
random.seed(42)
configs = Dict(json.loads(sys.argv[1])) if len(sys.argv) > 1 else Dict({'name': 'federated'})
print(configs)
logger = SQLiteLogger.new_instance('splitlearn_v2_1.sqlite', configs)
printer = logging.getLogger('federated')
# configs
rounds = configs.get('rounds', global_configs['rounds'])
epochs = configs.get('epochs', global_configs['epoch'])
batch = configs.get('batch', global_configs['batch'])
client_model = global_configs['client_model']
server_model = global_configs['server_model']
model = global_configs['model']
mnist = global_configs['train']
test_data = global_configs['test']
cluster_speeds = global_configs['cls_speeds']
out_clusters = global_configs['out_size']
lr_client = global_configs['lr_client']
lr_server = global_configs['lr_server']

double_clustered = get_clients(mnist, out_clusters, cluster_speeds, client_model, lr=lr_client,
                               id_model=model, dataset_path=global_configs['dt_tag'])

clients = splitlearn.clients1d(double_clustered)
clients_data = {i: clients[i].data for i in range(len(clients))}

trainer_manager = SeqTrainerManager()
trainer_params = TrainerParams(trainer_class=TorchTrainer, optimizer='sgd', epochs=epochs, batch_size=batch,
                               criterion='cel', lr=lr_server)
federated = FederatedLearning(
    trainer_manager=trainer_manager,
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=batch, criterion='cel'),
    client_selector=client_selectors.All(),
    trainers_data_dict=clients_data,
    initial_model=model,
    num_rounds=rounds,
)
FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]).attach(federated)
federated.add_subscriber(SQLoggerCustom(logger, clients))
# federated.add_subscriber(Resumable(IODict(f'./cached_models.cs'), key=f'b{hashed_args}'))
# federated.add_subscriber(RoundAccuracy(plot_ratio=1, plot_title='Round Accuracy 1'))
printer.info("----------------------")
printer.info(f"start federated genetics")
printer.info("----------------------")
federated.start()
