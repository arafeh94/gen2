import copy
import json
import logging
import torch
from apps.sequential_fed.s_core import warmups, pfed
from src.apis import lambdas, utils
from src.apis.utils import TimeCheckpoint
import configs
from configs import distributor
from src.federated.subscribers.sqlite_logger import SQLiteLogger

logger = logging.getLogger('seqfed')
utils.enable_logging()

train = configs.train
test = configs.test
base_model = configs.base_model
train_clients = distributor.distribute(train).map(lambdas.as_tensor)
run_acc = {}
run_time = {}
for cfid, config in configs.runs.items():
    db_logger = SQLiteLogger.new_instance('seqfed.sqlite', config)
    logger.info(f"running config: {config}")
    run_id = config.id if 'id' in config else None
    model = copy.deepcopy(base_model)
    method = config.method
    acc = []
    if method.startswith('seqop'):
        initial_weights, acc_loss, times, selector = warmups.sequential_warmup_op(
            model, config.wmp.rounds, train_clients, test, config.wmp.epochs, config.wmp.lr, config.wmp.selector,
            config)
        selection_history = json.dumps(selector.selection_history)
        for r, ((acc, loss), time) in enumerate(zip(acc_loss, times)):
            db_logger.log(r, acc=acc, loss=loss, time=time, selector=selection_history)
        logger.info(f"final avg: {acc_loss[-1]}")
        logger.info(f"history: {selector.selection_history}")
        logger.info(f"top10: {[(cid, train_clients[cid], val) for cid, val in selector.top_n(10).items()]}")
    elif method.startswith("warmup"):
        timer = TimeCheckpoint()
        train_data, initial_weights, test_acc, acc_loss = warmups.original_warmup(
            config.wmp.data_ratio, train_clients, test, model, config.wmp.epochs, config.wmp.lr)
        logger.info(f"final avg: {acc_loss}")
        logger.info(f"time taken: {timer.cp()}")
        for r, acc in enumerate(test_acc):
            db_logger.log(r, test_acc=acc / 100, acc=acc_loss[0], loss=acc_loss[1], time_taken=timer.cp(-1))
    elif method.startswith("ewc"):
        ewc_model, acc_loss, times, selector = warmups.ewc_warmup(
            train_clients, model, config.wmp.rounds, config.wmp.epochs, test, config.wmp.lr, config.wmp.weight,
            config.wmp.selector, config)
        initial_weights = ewc_model.model.state_dict()
        selection_history = json.dumps(selector.selection_history)
        for r, ((acc, loss), time) in enumerate(zip(acc_loss, times)):
            db_logger.log(r, acc=acc, loss=loss, time=time, selector=selection_history)
        logger.info(f"final avg: {acc_loss[-1]}")
        logger.info(f"history: {selector.selection_history}")
        logger.info(f"top10: {[(cid, train_clients[cid], val) for cid, val in selector.top_n(10).items()]}")
    else:
        initial_weights = model.state_dict()
    db_logger.close()
    torch.save(initial_weights, f'./weights/{config.id}.pt')
