import logging

from apps.donotuse.cluster_noniidness.core.modules import Cluster_TrainerProvider
from apps.main_split import dist
from apps.main_split.client import Client
from apps.main_split.models import MnistNet
from apps.donotuse.split_learning import Cluster
from src.apis import utils
from apps.splitfed.core import clusters
from src.data.data_loader import preload
from src.federated.components import aggregators, metrics, client_selectors
from src.federated.components.client_scanners import DefaultScanner
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger

utils.enable_logging()

id_tracker = 0

# total number of clusters is fixed to 5 ([1, 2], [3, 4], [5, 6], [7, 8], [9, 0])
# the max number of client in each cluster, use to avoid accuracy problem and cat. for.
cluster_limit = 0
# initialize clients, it should be replaced with weight divergence analysis.
# In this case, we have 20 clusters in each we have 'client_cluster_size' clients
client_cluster_size = 20

model = lambda: MnistNet(28 * 28, 32, 10)
clients_data = preload('mnist', dist.mnist_clustered(client_cluster_size, 600),
                       tag=f'clusters11{client_cluster_size}p{600}')
test_data = preload('mnist10k').as_tensor()
feds = []
c_clusters = clusters.generate(clients_data, model(), client_cluster_size, cluster_limit)
round_before_next_cluster = 1
rounds = 20
logger = logging.getLogger('main')

for index, cl in c_clusters.items():
    cl: Cluster
    # trainers configuration
    trainer_params = TrainerParams(
        trainer_class=Client,
        batch_size=50, epochs=10, optimizer='sgd',
        criterion='cel', lr=0.1)
    trainers_dict = FederatedLearning.on_device_trainer_data_builder(len(cl.clients))
    # fl parameters
    federated = FederatedLearning(
        trainer_manager=SeqTrainerManager(Cluster_TrainerProvider(cl.clients)),
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=50, criterion='cel'),
        client_scanner=DefaultScanner(trainers_dict),
        client_selector=client_selectors.All(),
        trainers_data_dict=trainers_dict,
        test_data=test_data,
        initial_model=model,
        num_rounds=50,
        desired_accuracy=0.99
    )

    # (subscribers)
    # federated.add_subscriber(TqdmLogger())
    federated.add_subscriber(FederatedLogger([Events.ET_ROUND_FINISHED]))
    # federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
    federated.init()
    feds.append(federated)

for i in range(rounds):
    logger.info(f'starting round: {i}')
    for j in range(len(feds)):
        logger.info(f'starting with fed: {j}')
        for z in range(round_before_next_cluster):
            feds[j].one_round()
        logger.info('finished, updating the model of the next')
        state_dict = feds[j].context.model.state_dict()
        feds[(j + 1) % len(feds)].context.model.load_state_dict(state_dict)
