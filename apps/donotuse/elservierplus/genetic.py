# mpiexec -n 4 python main_mpi.py
import logging
import sys

from apps.donotuse.dbs import distributors
from apps.genetics.src import initializer
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
    'epoch': 5,
    'batch': 999,
    'round': 500,
    'distributor': 'pipe2',
    'dataset': 'cifar10',
    'clients_ratio': 0.2,
    'learn_rate': 0.001,
    'tag': 'genetic-cifar104',
    'min': 600,
    'max': 600,
    'clients': 100,
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = preload(args.dataset, distributors.load(args)[args.distributor], tag=f'pip2-{args.dataset}')
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

    'nb_clusters': 20,
    'ga_max_iter': 50,
    'ga_r_cross': 0.05,
    'ga_r_mut': 0.1,
    'ga_c_size': 30,
    'ga_p_size': 200,
    'ga_min_fitness': 0,
    'ga_epochs': 50,

    'save_dir': 'pics',
}

initial_model = initializer.ga_module_creator(
    client_data, config['model'], max_iter=config['ga_max_iter'],
    r_cross=config['ga_r_cross'], r_mut=config['ga_r_mut'],
    c_size=config['ga_c_size'], p_size=config['ga_p_size'],
    clusters=config['nb_clusters'],
    desired_fitness=config['ga_min_fitness'], epoch=500, batch=50,
    saved_models=f'./saved_models_2_{config["ga_epochs"]}_{args.dataset}_{args.learn_rate}',
    lr=args.learn_rate
)

trainer_params = TrainerParams(trainer_class=TorchTrainer, optimizer='sgd', epochs=1,
                               batch_size=args.batch, criterion='cel', lr=args.learn_rate)
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
    accepted_accuracy_margin=0.01
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
