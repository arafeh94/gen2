import calendar
import logging
import random
import time

from apps.iidness.distributor import IidDistributor
from libs.model.cv.cnn import Cifar10Model
from src.apis import lambdas, utils
from src.apis.rw import IODict
from src.data.data_tools import iidness
from src.federated.subscribers.resumable import Resumable
from apps.slurm_jobs.paper_jobs import context
from src.data.data_loader import preload
from src.federated.components import aggregators, metrics, client_selectors
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.components.trainers import TorchTrainer
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.sqlite_logger import SQLiteLogger

args = context.args(1, 25, 2, 10, 0.01)
hashed_args = context.hashed(args)

log_file = f'{args.tag}_{hashed_args}.log'
utils.enable_logging(None)
logger = logging.getLogger('main')
distributors = [
    IidDistributor(100, 10, 600, 600, is_random_label_size=True),
    IidDistributor(100, 10, 600, 600, is_random_label_size=True),
    IidDistributor(100, 10, 600, 600, is_random_label_size=True),
    IidDistributor(100, 10, 600, 600, is_random_label_size=True),
    IidDistributor(100, 10, 600, 600, is_random_label_size=True),
    IidDistributor(100, 10, 600, 600, is_random_label_size=True),
]

for distributor in distributors:
    client_data = distributor.distribute(preload('cifar10'))
    client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
    idn = iidness(client_data.map(lambdas.as_list), 10)
    logger.info(idn)
    logger.info(client_data)
    prefix = f'{random.randint(0, 99999)}_{random.randint(0, 99999)}'
    config = {
        'batch_size': args.batch,
        'epochs': args.epochs,
        'clients_per_round': args.clients_ratio,
        'num_rounds': args.round,
        'desired_accuracy': 0.99,
        'model': lambda: Cifar10Model(),
        'lr': args.learn_rate,
        'id': hashed_args,
        'idn': idn,
        'title': 'cifar',
    }

    trainer_manager = SeqTrainerManager()
    trainer_params = TrainerParams(trainer_class=TorchTrainer, optimizer='sgd', epochs=config['epochs'],
                                   batch_size=config['batch_size'],
                                   criterion='cel', lr=config['lr'])
    federated = FederatedLearning(
        trainer_manager=trainer_manager,
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=config['batch_size'], criterion='cel'),
        client_selector=client_selectors.Random(config['clients_per_round']),
        trainers_data_dict=client_data,
        initial_model=config['model'],
        num_rounds=config['num_rounds'],
    )
    FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]).attach(federated)
    federated.add_subscriber(SQLiteLogger(str(calendar.timegm(time.gmtime())), f'cached_results.db', config))
    federated.add_subscriber(Resumable(IODict(f'./cached_models.cs'), key=f'b{prefix}_{hashed_args}'))
    logger.info("----------------------")
    logger.info(f"start federated genetics")
    logger.info("----------------------")
    federated.start()
