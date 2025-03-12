import logging
import sys

from src.federated.subscribers.sqlite_logger import SQLiteLogger

sys.path.append('../../')
from src.federated.subscribers.fed_plots import RoundAccuracy
from libs.model.linear.lr import LogisticRegression
from src.federated.components.client_scanners import DefaultScanner
from src.federated.events import Events
from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.timer import Timer
from src.data.data_distributor import UniqueDistributor, ShardDistributor
from src.data.data_loader import preload
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

client_data = preload('mnist', ShardDistributor(200, 2))
test_data = preload('mnist10k').as_tensor()

# trainers configuration
trainer_params = TrainerParams(
    trainer_class=trainers.TorchTrainer,
    batch_size=0, epochs=1, optimizer='sgd',
    criterion='cel', lr=0.1)

# fl parameters
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.F1(batch_size=0, criterion='cel'),
    test_data=test_data,
    client_scanner=DefaultScanner(client_data),
    client_selector=client_selectors.Random(0.1),
    trainers_data_dict=client_data,
    initial_model=lambda: LogisticRegression(784, 10),
    num_rounds=100,
    desired_accuracy=0.99
)

# (subscribers)
federated.add_subscriber(TqdmLogger())
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND, Timer.TRAINING]))
federated.add_subscriber(RoundAccuracy(plot_ratio=0))
federated.add_subscriber(SQLiteLogger('1', 'test.db'))

logger.info("------------------------")
logger.info("start federated learning")
logger.info("------------------------")
federated.start()
