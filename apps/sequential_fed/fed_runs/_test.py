import copy
import json
import logging
import random

import torch
from easydict import EasyDict as edict
from apps.sequential_fed.s_core import warmups, pfed
from apps.splitfed.models import MnistNet
from libs.model.cv.cnn import Cifar10Model
from libs.model.linear.lr_kdd import KDD_LR
from src.apis import lambdas, utils
from src.apis.extensions import TorchModel
from src.data.data_distributor import DirichletDistributor
from src.data.data_loader import preload
from src.federated.subscribers.sqlite_logger import SQLiteLogger

config = {
    'run_id': random.randint(100000, 999999),
    'method': 'seqop_ga',
    'wmp': {
        'selector': 'ga',
        'rounds': 50,
        'epochs': 500,
        'lr': 0.01,
        'cr': 10,
        'buffer': 'pop',
        'cls': 10,
    },
}

logger = logging.getLogger('seqfed')
utils.enable_logging()
db_logger = SQLiteLogger.new_instance('seqfed2.sqlite', config)
config = edict(config)
dist = DirichletDistributor(120, 10, 0.1)

cifar10 = preload('cifar10').map(lambdas.reshape((32, 32, 3))).map(lambdas.transpose((2, 0, 1))).shuffle()
train, test = cifar10.split(0.8)
train = train.as_tensor()
test = test.as_tensor()
model = Cifar10Model()
trainer = TorchModel(model)
trainer.train(train.batch(), epochs=500)
res = trainer.infer(test.batch())
print(res)
