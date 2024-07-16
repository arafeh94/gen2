import logging
import random
from pathlib import Path

from apps.iidness.distributor import IidDistributor
from libs.model.linear.lr import LogisticRegression
from src.apis import lambdas
from src.data.data_distributor import LabelDistributor
from src.data.data_loader import preload
from src.data.data_tools import iidness
from src.federated.components import trainers, aggregators, metrics, client_selectors
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.sqlite_logger import SQLiteLogger
from src.federated.subscribers.timer import Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
Path('results.db').unlink(missing_ok=True)
distributors = [
    IidDistributor(100, 10, 600, 600, is_random_label_size=True),
    IidDistributor(100, 10, 600, 600, is_random_label_size=True),
    IidDistributor(100, 10, 600, 600, is_random_label_size=True),
    IidDistributor(100, 10, 600, 600, is_random_label_size=True),
    IidDistributor(100, 10, 600, 600, is_random_label_size=True),
    IidDistributor(100, 10, 600, 600, is_random_label_size=True),
]

for distributor in distributors:
    client_data = distributor.distribute(preload('mnist'))
    idn = iidness(client_data.map(lambdas.as_list), 10)
    print(idn)
    trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=10_000, epochs=1, optimizer='sgd',
                                   criterion='cel', lr=0.1)
    federated = FederatedLearning(
        trainer_manager=SeqTrainerManager(),
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=50, criterion='cel'),
        client_selector=client_selectors.Random(0.1),
        trainers_data_dict=client_data,
        initial_model=lambda: LogisticRegression(28 * 28, 10),
        num_rounds=100,
        desired_accuracy=0.99
    )
    i = random.randint(0, 99999)
    j = random.randint(0, 99999)
    federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.add_subscriber(SQLiteLogger(f'{i}_{j}_{distributor.id()}', './results.db', {'iidness': idn}))
    federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))

    logger.info("----------------------")
    logger.info("start federated learning")
    logger.info("----------------------")
    federated.start()
