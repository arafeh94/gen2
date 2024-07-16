import copy
import json
import logging
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from easydict import EasyDict as edict
from apps.sequential_fed.s_core import warmups, pfed
from apps.splitfed.models import MnistNet
from libs.model.cv.cnn import Cifar10Model
from libs.model.cv.resnet import resnet56
from libs.model.linear.lr_kdd import KDD_LR
from src.apis import lambdas, utils
from src.data.data_distributor import DirichletDistributor
from src.data.data_loader import preload
from src.federated.subscribers.sqlite_logger import SQLiteLogger


class Cifar10CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.BatchNorm2d(256),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, xb):
        return self.network(xb)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def __main__():
    config = {
        'run_id': random.randint(100000, 999999),
        'method': 'seqop_ga',
        'wmp': {
            'selector': 'ga',
            'rounds': 400,
            'epochs': 20,
            'lr': 0.01,
            'cr': 10,
            'buffer': 'pop',
            'cls': 5,
        },
    }

    logger = logging.getLogger('seqfed')
    utils.enable_logging()
    db_logger = SQLiteLogger.new_instance('seqfed2.sqlite', config)
    config = edict(config)
    dist = DirichletDistributor(120, 10, 0.1)

    # cifar10 = preload('cifar10').shuffle()
    #
    # train, test = cifar10.split(0.8)
    # train = train.as_tensor()
    # test = test.as_tensor()
    # clients = dist.distribute(train)
    #
    # pickle.dump(clients, open('clients_flat.pkl', 'wb'))
    # pickle.dump(test, open('test_flat.pkl', 'wb'))
    # exit(1)

    clients = pickle.load(open('clients.pkl', 'rb'))
    test = pickle.load(open('test.pkl', 'rb'))

    method = config['method']
    # model = Cifar10Model()
    # model = resnet56(10, 3, 32)
    model = CNN()

    # initial_weights = model.state_dict()

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
        pickle.dump(initial_weights, open('initial_weights_batch.pkl', 'wb'))

    exit(1)
    fed_config = {'rounds': 10000, 'lr': 0.0001, 'epochs': 20, 'cr': 10}
    initial_weights = pickle.load(open('initial_weights.pkl', 'rb'))
    model.load_state_dict(initial_weights)
    federate = pfed.create_fl(clients, test, model, fed_config, db_logger)
    federate.start()
