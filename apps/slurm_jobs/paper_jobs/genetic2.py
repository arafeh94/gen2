import calendar
import logging
import time

from src.apis.rw import IODict
from src.federated.subscribers.resumable import Resumable
from apps.slurm_jobs.paper_jobs import context
from libs.model.cv.cnn import Cnn1D
from src.apis import lambdas
from src.data.data_loader import preload
from torch import nn
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.sqlite_logger import SQLiteLogger
from src.federated.components.trainers import TorchTrainer
from src.federated.protocols import TrainerParams
from src.federated.components import metrics, aggregators
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.subscribers.analysis import ClientSelectionCounter
from apps.genetics.src import initializer
from apps.genetics.src.selector import GeneticSelector

args = context.args()
hashed_args = context.hashed(args)

logging.basicConfig(filename=f'{args.tag}_{hashed_args}.log', filemode='w', datefmt='%H:%M:%S', level=logging.INFO)

logger = logging.getLogger('main')
logger.info('Generating Data --Started')
client_data = preload('fall_ar_by_client').map(lambdas.as_tensor)
logger.info('Generating Data --Ended')

config = {
    # federated learning configs
    'batch_size': args.batch,
    'epochs': args.epochs,
    'clients_per_round': args.clients_ratio,
    'num_rounds': args.round,
    'desired_accuracy': 0.99,
    'model': lambda: Cnn1D(15),
    'lr': args.learn_rate,
    'id': hashed_args,
    # genetic_configs
    'max_iter': 15,
    'r_cross': 0.05,
    'r_mut': 0.1,
    'c_size': args.chrm_size,
    'p_size': 200,
    'nb_clusters': args.nb_clusters,
    'desired_fitness': 0,
}

initiator, initial_model = initializer.ga_module_creator(
    client_data, config['model'], config, epoch=50,
    batch=config['batch_size'], lr=0.01)
client_selector = GeneticSelector(initiator, config['clients_per_round'], config)

trainer_manager = SeqTrainerManager()
trainer_params = TrainerParams(trainer_class=TorchTrainer, optimizer='sgd', epochs=config['epochs'],
                               batch_size=config['batch_size'], criterion='cel', lr=config['lr'])
federated = FederatedLearning(
    trainer_manager=trainer_manager,
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=config['batch_size'], criterion=nn.CrossEntropyLoss()),
    client_selector=client_selector,
    trainers_data_dict=client_data,
    initial_model=initial_model,
    num_rounds=config['num_rounds'],
    desired_accuracy=config['desired_accuracy'],
)

FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]).attach(federated)
federated.add_subscriber(SQLiteLogger(str(calendar.timegm(time.gmtime())), f'cached_results.db', config))
federated.add_subscriber(Resumable(IODict(f'./cached_models.cs'), key=f'b{hashed_args}'))
ClientSelectionCounter(save_dir=f'cls_{args.tag}_{hashed_args}.png').attach(federated)
client_selector.attach(federated)
logger.info("----------------------")
logger.info(f"start federated genetics")
logger.info("----------------------")
federated.start()
