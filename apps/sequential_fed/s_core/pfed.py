import copy

from src.apis.rw import IODict
from src.apis.utils import TimeCheckpoint
from src.federated.components import trainers, aggregators, metrics, client_selectors
from src.federated.components.client_scanners import DefaultScanner
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.events import Events, FederatedSubscriber
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import TqdmLogger, FederatedLogger
from src.federated.subscribers.resumable import Resumable
from src.federated.subscribers.sqlite_logger import SQLiteLogger
from src.federated.subscribers.timer import Timer
from src.federated.subscribers.wandb_logger import WandbLogger


class SQLLoggerWithTimer(FederatedSubscriber):

    def __init__(self, sql_logger: SQLiteLogger):
        super().__init__()
        self.sql_logger = sql_logger
        self.timer = TimeCheckpoint()

    def on_round_start(self, params):
        self.timer.checkpoint()

    def on_round_end(self, params):
        context: FederatedLearning.Context = params['context']
        last_record: dict = context.history[context.round_id]
        self.sql_logger.log(context.round_id, acc=last_record['acc'], loss=last_record['loss'],
                            time=self.timer.checkpoint(), selector='')


def create_fl(client_data, test, model, config, sql_logger):
    # trainers configuration
    trainer_params = TrainerParams(
        trainer_class=trainers.TorchTrainer,
        batch_size=100, epochs=config['epochs'], optimizer='sgd',
        criterion='cel', lr=config['lr'])

    # fl parameters
    federated = FederatedLearning(
        trainer_manager=SeqTrainerManager(),
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=0, criterion='cel'),
        client_scanner=DefaultScanner(client_data),
        client_selector=client_selectors.Random(config['cr']),
        trainers_data_dict=client_data,
        test_data=test,
        # accepted_accuracy_margin=0.001,
        initial_model=lambda: copy.deepcopy(model),
        num_rounds=config['rounds'],
        desired_accuracy=0.99
    )

    # (subscribers)
    federated.add_subscriber(TqdmLogger())
    federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
    federated.add_subscriber(SQLLoggerWithTimer(sql_logger))
    federated.add_subscriber(Resumable(IODict('./resumable.io'), save_ratio=50))
    # if 'wandb' not in config or ('wandb' in config and config['wandb']):
    #     federated.add_subscriber(WandbLogger(config=config, id=id))
    return federated
