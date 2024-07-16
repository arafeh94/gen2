import logging

from apps.donotuse.genetic_selectors.algo import initializer
from apps.genetic_selectors_v2.new_genetic_selectors.algo.selector import GeneticSelector
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


logging.basicConfig(filename='logs.txt', filemode='a', datefmt='%H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger('main')
db_name = './results.db'
experiment_name = f'test_2'
logger.info('Generating Data --Started')
client_data = preload('fall_ar_by_client').map(lambdas.as_tensor)
logger.info('Generating Data --Ended')

configs = [{
    # federated learning configs
    'batch_size': 80_000,
    'epochs': 25,
    'clients_per_round': 5,
    'num_rounds': 500,
    'desired_accuracy': 0.99,
    'model': lambda: Cnn1D(15),
    # genetic_configs
    'max_iter': 10,
    'r_cross': 0.05,
    'r_mut': 0.1,
    'c_size': 8,
    'p_size': 200,
    'nb_clusters': 1,
    'desired_fitness': 0,
}, {
    # federated learning configs
    'batch_size': 80_000,
    'epochs': 1,
    'clients_per_round': 5,
    'num_rounds': 500,
    'desired_accuracy': 0.99,
    'model': lambda: Cnn1D(15),
    # genetic_configs
    'max_iter': 10,
    'r_cross': 0.05,
    'r_mut': 0.1,
    'c_size': 8,
    'p_size': 200,
    'nb_clusters': 1,
    'desired_fitness': 0,
}, ]

for config in configs:
    initiator, initial_model = initializer.ga_module_creator(
        client_data, config['model'], config, epoch=50,
        batch=config['batch_size'], lr=0.01)
    client_selector = GeneticSelector(initiator, config['clients_per_round'], config)

    trainer_manager = SeqTrainerManager()
    trainer_params = TrainerParams(trainer_class=TorchTrainer, optimizer='sgd', epochs=config['epochs'],
                                   batch_size=config['batch_size'], criterion='cel', lr=0.01)
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
    SQLiteLogger(f'{experiment_name}', db_name).attach(federated)
    ClientSelectionCounter(save_dir='plots/').attach(federated)
    client_selector.attach(federated)
    logger.info("----------------------")
    logger.info(f"start federated genetics")
    logger.info("----------------------")
    federated.start()
