import logging

import numpy as np
from matplotlib import pyplot as plt

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

d = preload('femnist').as_numpy()
d = d.x[100]
plt.imshow(np.reshape(d, (28, 28)))
plt.show()

d = preload('mnist').as_numpy()
d = d.x[0]
plt.imshow(np.reshape(d, (28, 28)))
plt.show()