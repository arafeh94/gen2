import logging
import sys

from apps.donotuse.framework_experiments import distributors
from src.federated.subscribers.analysis import ShowWeightDivergence

sys.path.append("../../../")

from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.timer import Timer

from src.data.data_loader import preload
from torch import nn
from src.apis import federated_args
from libs.model.linear.lr import LogisticRegression
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

args = federated_args.FederatedArgs({
    'epoch': 25,
    'batch': 50,
    'round': 200,
    'distributor': 2,
    'dataset': 'mnist',
    'clients_ratio': 0.999,
    'learn_rate': 0.01,
    'tag': 'mnist',
    'min': 600,
    'max': 600,
    'clients': 10,
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = preload(args.dataset, distributors.load(args)[args.distributor])
print(args.distributor)
print(client_data)
logger.info('Generating Data --Ended')

initial_model = LogisticRegression(28 * 28, 10)

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=args.batch, epochs=args.epoch,
                               optimizer='sgd', criterion='cel', lr=args.learn_rate)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=args.batch, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(args.clients_ratio),
    trainers_data_dict=client_data,
    initial_model=lambda: initial_model,
    num_rounds=args.round,
    desired_accuracy=1,
)

federated.add_subscriber(TqdmLogger())
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(ShowWeightDivergence(save_dir=f"./{args.tag}", plot_type='linear'))

# federated.add_subscriber(BandwidthTracker())

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
