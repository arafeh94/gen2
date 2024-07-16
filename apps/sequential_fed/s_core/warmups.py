import copy
import logging
import time
from typing import Union

from torch import nn
from tqdm import tqdm

from apps.genetics.src import alg_genetic
from apps.sequential_fed.s_core import cluster_creator, selectors
from apps.sequential_fed.s_core.selectors import SeqGAClientSelector
from src.apis import federated_tools, lambdas, utils
from src.apis.ewc import ElasticWeightConsolidation
from src.apis.extensions import TorchModel
from src.apis.utils import TimeCheckpoint
from src.data.data_container import DataContainer

logger = logging.getLogger('warmups')


def sequential_warmup_op(model, rounds, train_clients: dict, test_dataset: DataContainer, epochs, lr, selector_id,
                         configs):
    total_accs = []
    times = []
    timer = TimeCheckpoint()
    trainer = TorchModel(copy.deepcopy(model))
    last_round_trainers = []
    selector = selectors.create(selector_id, train_clients, configs)
    selector.build(base_model=copy.deepcopy(model)) if selector_id == 'ga' else selector.build()
    for k in tqdm(range(rounds), desc="Training Rounds"):
        trainer_old: Union[TorchModel, None] = None
        while selector.next():
            trainer_id = selector.get_id()
            trainer.train(train_clients[trainer_id].batch(50), lr=lr, epochs=epochs, verbose=0)
            if k != rounds - 1 and trainer_old is not None:
                trainer.dilute(trainer_old, 6)
            trainer_old = trainer.copy()
            if k == rounds - 1:
                last_round_trainers.append(trainer.copy())
        selector.reset()
        test_acc = trainer.infer(test_dataset.batch(), verbose=0)
        total_accs.append(test_acc)
        logger.info(f"Test Accuracy: {test_acc}")
        times.append(timer.cp())
    models_states = {}
    models_sizes = {}
    for i in range(len(last_round_trainers)):
        models_states[i] = last_round_trainers[i].model.state_dict()
        models_sizes[i] = 1
    weights = federated_tools.aggregate(models_states, models_sizes)
    trainer.load(weights)
    total_accs.append(trainer.infer(test_dataset.batch(), verbose=0))
    times.append(timer.cp())
    return weights, total_accs, times, selector


def original_warmup(warmup_ratio, train_data, test_data, model, epochs=500, lr=0.0001):
    # collect data from clients
    warmup_data = train_data.map(lambda ci, dc: dc.shuffle(42).split(warmup_ratio)[0]).reduce(lambdas.dict2dc)
    # redistribute collected data to all clients
    train_data = train_data.map(lambdas.as_numpy).map(lambda cid, dt: dt.concat(warmup_data))
    initial_model = TorchModel(copy.deepcopy(model))
    weights, test_acc = initial_model.train(warmup_data.as_tensor().batch(), epochs=epochs, verbose=1, lr=lr)
    acc_loss = initial_model.infer(test_data.batch(), verbose=0)
    return train_data.map(lambdas.as_tensor), weights, test_acc, acc_loss


def ewc_warmup(clients, model, rounds, wp_epochs, test_dataset, lr=0.0001, weight=0.1, selector_id='', config={}):
    timer = TimeCheckpoint()
    ewc = ElasticWeightConsolidation(copy.deepcopy(model), nn.CrossEntropyLoss(), lr=lr, weight=weight)
    acc = []
    times = []
    selector = selectors.create(selector_id, clients, config)
    selector.build(base_model=copy.deepcopy(model)) if selector_id == 'ga' else selector.build()
    for _ in tqdm(range(rounds), desc="Training Rounds"):
        while selector.next():
            trainer_id = selector.get_id()
            ewc.train(clients[trainer_id], epochs=wp_epochs)
        selector.reset()
        acc.append(federated_tools.infer(ewc.model, test_dataset.batch()))
        times.append(timer.cp())
    return ewc, acc, times, selector
