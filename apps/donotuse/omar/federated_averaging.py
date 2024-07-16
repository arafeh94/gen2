import logging
import random
import sys

from src.federated.subscribers.analysis import ClientSelectionCounter
from src.federated.subscribers.fed_plots import RoundAccuracy
from src.federated.subscribers.sqlite_logger import SQLiteLogger

sys.path.append('../../../')
from libs.model.linear.lr import LogisticRegression
from src.federated.components.client_scanners import DefaultScanner
from src.federated.events import Events
from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.timer import Timer
from src.data.data_distributor import DirichletDistributor, ShardDistributor
from src.data.data_loader import preload
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

clients = preload('mnist', ShardDistributor(300, 2))

# trainers configuration
trainer_params = TrainerParams(
    trainer_class=trainers.TorchTrainer,
    batch_size=500, epochs=5, optimizer='sgd',
    criterion='cel', lr=0.001)
selector = client_selectors.ClusterSelector(10, 3)

# fl parameters
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=500, criterion='cel'),
    client_scanner=DefaultScanner(clients),
    client_selector=selector,
    trainers_data_dict=clients,
    initial_model=lambda: LogisticRegression(28 * 28, 10),
    num_rounds=10,
    desired_accuracy=0.99
)

# (subscribers)
selector.attach(federated)
federated.add_subscriber(TqdmLogger())
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer())
federated.add_subscriber(SQLiteLogger('kdd_res', './tst/res.db'))
federated.add_subscriber(ClientSelectionCounter('./selected_clients.png'))
federated.add_subscriber(RoundAccuracy(plot_ratio=0))

logger.info("------------------------")
logger.info("start federated learning")
logger.info("------------------------")
federated.start()