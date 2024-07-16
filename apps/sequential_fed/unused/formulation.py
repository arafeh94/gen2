import logging
import random
import sys

from apps.main_split.models import MnistNet
from src.apis import lambdas
from src.apis.extensions import Dict
from src.data.data_container import DataContainer
from src.federated.subscribers.fed_plots import FinalAccuracyPlot, RoundAccuracy, EMDWeightDivergence, WeightPlot

sys.path.append('../../../')
from libs.model.linear.lr import LogisticRegression
from src.federated.components.client_scanners import DefaultScanner
from src.federated.events import Events
from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.timer import Timer
from src.data.data_distributor import ShardDistributor, LabelDistributor, Distributor
from src.data.data_loader import preload
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

client_data = preload('mnist', LabelDistributor(10, 1, 10, 150))

# trainers configuration
trainer_params = TrainerParams(
    trainer_class=trainers.TorchTrainer,
    batch_size=50, epochs=50, optimizer='sgd',
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
    num_rounds=1,
    desired_accuracy=0.99
)

# (subscribers)
federated.add_subscriber(TqdmLogger())
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
federated.add_subscriber(WeightPlot(show_plot=1))

logger.info("------------------------")
logger.info("start federated learning")
logger.info("------------------------")
federated.start()
