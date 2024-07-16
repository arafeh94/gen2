# mpiexec -n 4 python main_mpi.py
import logging
import sys

from apps.donotuse.dbs import distributors
from src.federated.subscribers.fed_plots import EMDWeightDivergence
from src.federated.subscribers.sqlite_logger import SQLiteLogger

sys.path.append("../../../")

from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.timer import Timer
from libs.model.cv.cnn import Cifar10Model
from src.apis import lambdas, federated_args, utils
from src.data.data_loader import preload
from libs.model.linear.lr import LogisticRegression
from src.federated.components.trainers import TorchTrainer
from src.federated.protocols import TrainerParams
from src.federated.components import metrics, client_selectors, aggregators
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.components.trainer_manager import SeqTrainerManager

args = federated_args.FederatedArgs({
    'epoch': 1,
    'batch': 999,
    'round': 500,
    'distributor': 'pipe',
    'dataset': 'cifar10',
    'clients_ratio': 0.1,
    'learn_rate': 0.001,
    'tag': 'base-cifar10',
    'min': 600,
    'max': 600,
    'clients': 100,
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = preload(args.dataset, distributors.load(args)[args.distributor], tag=f'pip-{args.dataset}-mnist')
print(client_data)
logger.info('Generating Data --Ended')

if args.dataset == 'mnist':
    c_model = LogisticRegression(28 * 28, 10)
elif args.dataset == 'cifar10':
    client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
    c_model = Cifar10Model()
else:
    c_model = LogisticRegression(28 * 28, 10)

config = {
    'batch_size': args.batch,
    'epochs': args.epoch,
    'clients_per_round': args.clients_ratio,
    'num_rounds': args.round,
    'desired_accuracy': 0.99,
    'model': lambda: c_model,
    'save_dir': 'pics',
}

initial_model = lambda: c_model

trainer_params = TrainerParams(trainer_class=TorchTrainer, optimizer='sgd', epochs=args.epoch,
                               batch_size=config['batch_size'], criterion='cel', lr=args.learn_rate)
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=config['batch_size'], criterion='cel'),
    client_selector=client_selectors.Random(config['clients_per_round']),
    trainers_data_dict=client_data,
    initial_model=initial_model,
    num_rounds=config['num_rounds'],
    desired_accuracy=config['desired_accuracy'],
    # accepted_accuracy_margin=0.01
)

federated.add_subscriber(TqdmLogger())
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(EMDWeightDivergence(show_plot=False))
federated.add_subscriber(SQLiteLogger(utils.hash_string(str(args)), db_path='res.db', config=args))

logger.info("----------------------")
logger.info(f"start federated 1")
logger.info("----------------------")
federated.start()
