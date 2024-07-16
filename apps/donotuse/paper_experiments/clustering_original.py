import copy
import logging
import sys
from datetime import datetime

from src.apis.extensions import Dict
from src.federated.subscribers.sqlite_logger import SQLiteLogger

sys.path.append("../../../")
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

from src.apis.utils import fed_avg
from src.federated.subscribers.logger import FederatedLogger

from libs.model.cv.cnn import Cifar10Model
from libs.model.linear.lr import LogisticRegression
from src.apis import files, lambdas, federated_args, utils, federated_tools
from src.data.data_distributor import LabelDistributor
from src.data.data_loader import preload
from src.federated.components import client_selectors
from src.federated.components.aggregators import AVGAggregator
from src.federated.components.metrics import AccLoss
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.components.trainers import TorchTrainer, TorchChunkTrainer
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams

args = federated_args.FederatedArgs({
    'epoch': 25,
    'batch': 50,
    'round': 50,
    'shard': 2,
    'dataset': 'mnist',
    'clients_ratio': 0.1,
    'learn_rate': 0.1,
    'tag': 'cluster',
    'min': 600,
    'max': 600,
    'clients': 100,
    'timestamp': datetime.now()
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = preload(args.dataset, LabelDistributor(args.clients, args.shard, args.min, args.max))
logger.info('Generating Data --Ended')

if args.dataset == 'cifar10':
    client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))


def create_model():
    if args.dataset == 'mnist':
        return LogisticRegression(28 * 28, 10)
    elif args.dataset == 'cifar10':
        return Cifar10Model()
    else:
        return LogisticRegression(28 * 28, 10)


initial_model = create_model()

config = {
    'batch_size': args.batch,
    'epochs': args.epoch,
    'clients_per_round': args.clients_ratio,
    'num_rounds': args.round,
    'desired_accuracy': 0.99,
    'nb_clusters': 10,
    'model': lambda: initial_model,

    'clustering_num_rounds': 5,
    'linkage': 'complete',
    'clustering_measuring_algo': 'cosine',
    'update_data_ration': 0.05,
    'clustering_data_ratio': 0.05,
}


def partition(per):
    part_data = Dict({})
    for client_id, data in client_data.items():
        part, rest = data.shuffle(47).split(per)
        part_data[client_id] = part.as_tensor()
        client_data[client_id] = rest.as_tensor()
    return part_data


cluster_data = partition(config['clustering_data_ratio'])
update_data = partition(config['update_data_ration'])
test_data = Dict(partition(0.2)).reduce(lambdas.dict2dc).as_tensor()

federated_learning = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=TrainerParams(trainer_class=TorchChunkTrainer, batch_size=config['batch_size'], epochs=5,
                                 criterion='cel', optimizer='sgd', lr=args.learn_rate),
    num_rounds=config['clustering_num_rounds'],
    client_selector=client_selectors.All(),
    desired_accuracy=0.99,
    train_ratio=0.8,
    metrics=AccLoss(config['batch_size'], 'cel'),
    aggregator=AVGAggregator(),
    initial_model=config['model'],
    trainers_data_dict=cluster_data,
)

federated_learning.add_subscriber(FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))

global_model = federated_learning.start()
w = global_model.state_dict()
clients_diff_weights = {}
for client_id, data in update_data.items():
    model_copy = copy.deepcopy(global_model)
    federated_tools.train(model_copy, data.batch(config['batch_size']), epochs=5, lr=args.learn_rate)
    wc = model_copy.state_dict()
    delta = utils.flatten_weights(w) - utils.flatten_weights(wc)
    clients_diff_weights[client_id] = delta

cluster = AgglomerativeClustering(n_clusters=config['nb_clusters'], affinity=config['clustering_measuring_algo'],
                                  linkage=config['linkage'])
cluster_dist = cluster.fit_predict(list(clients_diff_weights.values()))
client_ids = list(clients_diff_weights.keys())
clustered_clients = defaultdict(list)
for cluster_id, client_id in zip(cluster_dist, client_ids):
    clustered_clients[cluster_id].append(client_id)
logger.info(cluster_dist)

clustered_federated = {}
for cluster, client_ids in clustered_clients.items():
    logger.info(f'cluster:{cluster}-clients:{client_ids}')
    federated = FederatedLearning(
        trainer_manager=SeqTrainerManager(),
        trainer_config=TrainerParams(trainer_class=TorchTrainer, batch_size=args.batch, epochs=args.epoch,
                                     criterion='cel', optimizer='sgd', lr=args.learn_rate),
        num_rounds=config['num_rounds'],
        client_selector=client_selectors.Random(args.clients_ratio),
        desired_accuracy=0.99,
        train_ratio=0.8,
        # test_data=test_data,
        metrics=AccLoss(config['batch_size'], 'cel'),
        aggregator=AVGAggregator(),
        initial_model=lambda: global_model,
        trainers_data_dict=utils.dict_select(client_ids, client_data),
    )

    federated.add_subscriber(FederatedLogger([Events.ET_ROUND_FINISHED]))
    federated.init()
    clustered_federated[cluster] = federated

sqlogger = SQLiteLogger(utils.hash_string(str(args)), tag=str(args))
sqlogger.init()
finished_tasks = [False] * len(clustered_federated)
round_id = 0
while not all(finished_tasks):
    for index, (cluster_id, federated) in enumerate(clustered_federated.items()):
        logger.info(f'{cluster_id}')
        finished_tasks[index] = federated.one_round()
    one_round_context = [f.context for f in clustered_federated.values()]
    avg_acc, avg_loss = fed_avg(one_round_context)
    sqlogger.log(round_id, acc=list(avg_acc.values())[-1], loss=list(avg_loss.values())[-1])
    round_id += 1

runs = [f.context for f in clustered_federated.values()]
avg_acc, avg_loss = fed_avg(runs)
files.accuracies.append(str(args), list(avg_acc.values()))
