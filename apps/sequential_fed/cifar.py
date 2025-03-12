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
from src.data.data_distributor import DirichletDistributor
from src.data.data_loader import preload
from configs import distributor, fed_config
from src.federated.subscribers.sqlite_logger import SQLiteLogger

config = {
    'run_id': random.randint(100000, 999999),
    # 'method': 'seqop_ga',
    'method': 'none',
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
clients = dist.distribute(train)
method = config['method']
model = Cifar10Model()
initial_weights = model.state_dict()

if method.startswith('seqop'):
    initial_weights, acc_loss, times, selector = warmups.sequential_warmup_op(
        model, config.wmp.rounds, clients, test, config.wmp.epochs, config.wmp.lr, config.wmp.selector,
        config)
    selection_history = json.dumps(selector.selection_history)
    for r, ((acc, loss), time) in enumerate(zip(acc_loss, times)):
        db_logger.log(r - config.wmp.rounds, acc=acc, loss=loss, time=time, selector=selection_history)
    logger.info(f"final avg: {acc_loss[-1]}")
    logger.info(f"history: {selector.selection_history}")
    logger.info(f"top10: {[(cid, clients[cid], val) for cid, val in selector.top_n(10).items()]}")

fed_config = {'rounds': 300, 'lr': 0.00001, 'epochs': 50, 'cr': 10}
model.load_state_dict(initial_weights)
federate = pfed.create_fl(clients, test, model, fed_config, db_logger)
federate.start()
