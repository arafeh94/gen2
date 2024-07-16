# mpiexec -n 4 python main_mpi.py
import logging
import sys
from os.path import dirname
from pathlib import Path

from torch import nn

from apps.flsim.src.client_selector import RLSelector
from apps.flsim.src.initializer import rl_module_creator
from apps.donotuse.genetic_selectors import distributor
from apps.donotuse.genetic_selectors.algo import initializer
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.sqlite_logger import SQLiteLogger

sys.path.append(dirname(__file__) + '../')

from libs.model.linear.lr import LogisticRegression
from src.federated.components.trainers import CPUTrainer
from src.federated.protocols import TrainerParams
from src.federated.components import metrics, client_selectors, aggregators
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

batch_size = 50
epochs = 1
clients_per_round = 10
num_rounds = 100
model = lambda: LogisticRegression(28 * 28, 10)
db_name = 'res7.db'
logger.info('Generating Data --Started')
client_data = distributor.get_distributed_data()
logger.info('Generating Data --Ended')

configs = {
    'ga': {
        'batch_size': batch_size,
        'epochs': epochs,
        'clients_per_round': clients_per_round,
        'num_rounds': num_rounds,
        'desired_accuracy': 0.99,
        'model': model,
        'ga_max_iter': 10,
        'ga_r_cross': 0.05,
        'ga_r_mut': 0.1,
        'ga_c_size': 10,
        'ga_p_size': 200,
        'nb_clusters': 5,
        'ga_min_fitness': 0.45,
    },
    # 'normal': {
    #     'batch_size': batch_size,
    #     'epochs': epochs,
    #     'clients_per_round': clients_per_round,
    #     'num_rounds': num_rounds,
    #     'desired_accuracy': 0.99,
    #     'model': model,
    # },
    'clustered': {
        'batch_size': batch_size,
        'epochs': epochs,
        'clients_per_round': clients_per_round,
        'num_rounds': num_rounds,
        'desired_accuracy': 0.99,
        'model': model,
        'c_size': 10,
        'nb_clusters': 5,
    },
    # 'rl': {
    #     'batch_size': batch_size,
    #     'epochs': epochs,
    #     'clients_per_round': clients_per_round,
    #     'num_rounds': num_rounds,
    #     'desired_accuracy': 0.99,
    #     'model': model,
    # }

}

Path(db_name).unlink(missing_ok=True)
for name, config in configs.items():

    initial_model = config['model']
    if name == 'ga':
        initial_model = initializer.ga_module_creator(client_data, initial_model, max_iter=config['ga_max_iter'],
                                                      r_cross=config['ga_r_cross'], r_mut=config['ga_r_mut'],
                                                      c_size=config['ga_c_size'], p_size=config['ga_p_size'],
                                                      clusters=config['nb_clusters'], epoch=50,
                                                      desired_fitness=config['ga_min_fitness'])
    elif name == 'clustered':
        initial_model = initializer.cluster_module_creator(client_data, initial_model, config['nb_clusters'],
                                                           config['c_size'])
    elif name == 'rl':
        initial_model0 = initial_model()
        booted_model, client_director, w_first = rl_module_creator(client_data, initial_model0)
        client_selector = RLSelector(config['clients_per_round'], client_director, w_first)
        initial_model = booted_model

    trainer_manager = SeqTrainerManager()
    trainer_params = TrainerParams(trainer_class=CPUTrainer, optimizer='sgd', epochs=epochs, batch_size=batch_size,
                                   criterion='cel', lr=0.1)
    federated = FederatedLearning(
        trainer_manager=trainer_manager,
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=config['batch_size'], criterion=nn.CrossEntropyLoss(), device='cpu'),
        client_selector=client_selectors.Random(config['clients_per_round']),
        trainers_data_dict=client_data,
        initial_model=initial_model,
        num_rounds=config['num_rounds'],
        desired_accuracy=config['desired_accuracy'],
        accepted_accuracy_margin=0.05
    )

    federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.add_subscriber(SQLiteLogger(f'genetic_cluster_test_{name}', db_name))
    logger.info("----------------------")
    logger.info(f"start federated {name}")
    logger.info("----------------------")
    federated.start()
