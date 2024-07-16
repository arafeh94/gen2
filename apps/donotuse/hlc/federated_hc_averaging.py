import logging
import statistics
import sys
from collections import defaultdict

from apps.donotuse.hlc import apis
from apps.donotuse.hlc.apis import SaveClientsModels
from src.apis import utils
from src.apis.extensions import Dict, TorchModel

sys.path.append('../../../')
from libs.model.linear.lr import LogisticRegression
from src.federated.components.client_scanners import DefaultScanner
from src.federated.events import Events
from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.timer import Timer
from src.data.data_distributor import LabelDistributor
from src.data.data_loader import preload
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

client_data = preload('mnist', LabelDistributor(100, 2, 600, 600))
test_data = preload('mnist10k').as_tensor()
warmup_rounds = 2
training_rounds = 3
total_selection = 20

cached_models = SaveClientsModels()
# trainers configuration
trainer_params = TrainerParams(
    trainer_class=trainers.TorchTrainer,
    batch_size=50, epochs=10, optimizer='sgd',
    criterion='cel', lr=0.1)

# fl parameters
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion='cel'),
    client_scanner=DefaultScanner(client_data),
    client_selector=client_selectors.All(),
    trainers_data_dict=client_data,
    initial_model=lambda: LogisticRegression(28 * 28, 10),
    num_rounds=warmup_rounds,
    desired_accuracy=0.99
)

# (subscribers)
federated.add_subscriber(TqdmLogger())
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
cached_models.attach(federated)

logger.info("------------------------")
logger.info("start federated learning")
logger.info("------------------------")
federated.start()

# I have the weights of each clients
clients_weights = cached_models.clients_weights
clusters_clients = utils.hc_clustering(clients_weights, 5)
clusters = defaultdict(list)
for cid, cluster_id in clusters_clients.items():
    clusters[cluster_id].append(cid)
clusters_federations = {}
selection_per_round = apis.get_nearly_equal_numbers(total_selection, len(clusters))
print(selection_per_round)
for c in clusters:
    cluster_clients_data = Dict(client_data).select(clusters[c])
    trainer_params = TrainerParams(
        trainer_class=trainers.TorchTrainer,
        batch_size=50, epochs=10, optimizer='sgd',
        criterion='cel', lr=0.1)
    federated = FederatedLearning(
        trainer_manager=SeqTrainerManager(),
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=50, criterion='cel'),
        client_scanner=DefaultScanner(cluster_clients_data),
        client_selector=client_selectors.Random(selection_per_round[c]),
        trainers_data_dict=cluster_clients_data,
        initial_model=lambda: LogisticRegression(28 * 28, 10),
        num_rounds=training_rounds,
        desired_accuracy=0.99
    )
    federated.add_subscriber(TqdmLogger())
    federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
    clusters_federations[c] = federated

for fe_id, fed in clusters_federations.items():
    fed.start()

cluster_models = {}
for cf in clusters_federations:
    cluster_models[cf] = TorchModel(clusters_federations[cf].context.model)


def global_infer(batched_data, all_clusters):
    test_results = {}
    for cmid in all_clusters:
        test = all_clusters[cmid].infer(batched_data, device='cuda')
        test_results[cmid] = test[0]
    return test_results


def global_infer_max(data, all_clusters):
    all_res = []
    for i in range(len(data)):
        test_results = {}
        for cmid in all_clusters:
            test = all_clusters[cmid].infer(data.select([i]).batch(0), device='cuda')
            test_results[cmid] = test[0]
        all_res.append(max(test_results.values()))
    return statistics.mean(all_res)


print(global_infer_max(test_data.select(range(20)), cluster_models))
