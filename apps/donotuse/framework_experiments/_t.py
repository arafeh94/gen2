import logging

from src.apis import utils
from src.apis.fed_sqlite import FedDB
from src.data.data_distributor import ShardDistributor, DirichletDistributor
from src.data.data_loader import preload
from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.resumable import Resumable
from src.federated.subscribers.timer import Timer

from libs.model.cv.cnn import Cifar10Model, CNN_OriginalFedAvg
from src.data.data_distributor import ShardDistributor, LabelDistributor, DirichletDistributor

from libs.model.cv.resnet import ResNet, resnet56
from src.data.data_loader import preload
from typing import Callable
from torch import nn
from src.apis import lambdas, files, federated_args, utils
from src.apis.extensions import TorchModel
from libs.model.linear.lr import LogisticRegression
from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated import subscribers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager, SharedTrainerProvider

# client_data = preload('mnist', DirichletDistributor(100, 10, 10))
# client_data = preload('femnist', ShardDistributor(500, 2))
# print(len(client_data))
utils.enable_logging()

client_data = preload('cifar10', ShardDistributor(300, int(5)))
client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2))).map(
    lambdas.as_numpy)
print(client_data)
initial_model = Cifar10Model()
trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=1,
                               optimizer='sgd', criterion='cel', lr=0.1)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(10),
    trainers_data_dict=client_data,
    initial_model=lambda: initial_model,
    num_rounds=500,
    desired_accuracy=1,
)
logging.log(1, 'samira')
federated.start()
