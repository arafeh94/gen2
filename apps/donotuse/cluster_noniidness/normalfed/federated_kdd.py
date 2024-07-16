import logging

from torch import nn

from apps.main_split import dist
from apps.main_split.models import MnistNet
from src.apis import lambdas
from src.data import data_loader
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.linear.lr import LogisticRegression
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.subscribers.analysis import ClientSelectionCounter
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer
from src.federated.subscribers.wandb_logger import WandbLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
all_data = preload('fekdd_train')
train_data, test_data = all_data.split(.8)
# clients_data = dist.clustered(20, 600).distribute(train_data).map(lambdas.as_tensor).filter(lambdas.empty)
clients_data = ShardDistributor(50, 4).distribute(train_data).map(lambdas.as_tensor).filter(lambdas.empty)
test_data = test_data.as_tensor()
logger.info('Generating Data --Ended')

trainer_config = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=100, epochs=1, optimizer='sgd',
                               criterion='cel', lr=0.01)
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_config,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=100, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(0.3),
    trainers_data_dict=clients_data,
    test_data=test_data,
    # initial_model=lambda: LogisticRegression(41, 23),
    initial_model=lambda: MnistNet(41, 32, 23),
    num_rounds=1000,
    desired_accuracy=0.99
)

federated.add_subscriber(FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
federated.add_subscriber(ClientSelectionCounter())
federated.add_subscriber(WandbLogger({'title': 'fed_kdd_normal', 'model': 'mnistnet', 'params': trainer_config,
                                      'selection': 'all', 'dist': 'shard4'}))

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
