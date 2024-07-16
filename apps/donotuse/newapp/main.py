import logging

from libs.model.linear.lr import LogisticRegression
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from src.federated.components import trainers, aggregators, metrics, client_selectors
from src.federated.components.client_scanners import DefaultScanner
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.fed_plots import RoundAccuracy
from src.federated.subscribers.logger import TqdmLogger, FederatedLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

shards = ShardDistributor(100, 5)
clients_data = preload('mnist', shards)

# trainers configuration
trainer_params = TrainerParams(
    trainer_class=trainers.TorchTrainer,
    batch_size=50, epochs=1, optimizer='sgd',
    criterion='cel', lr=0.1)

# fl parameters
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion='cel'),
    client_scanner=DefaultScanner(clients_data),
    client_selector=client_selectors.Random(0.1),
    trainers_data_dict=clients_data,
    initial_model=lambda: LogisticRegression(28 * 28, 10),
    num_rounds=20,
    desired_accuracy=0.99
)

federated.add_subscriber(TqdmLogger())
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(RoundAccuracy())

federated.start()
