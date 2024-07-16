import copy
from typing import Union
from tqdm import tqdm
from src.apis import federated_tools, lambdas, utils
from src.apis.extensions import TorchModel
from src.apis.utils import TimeCheckpoint
from src.data.data_container import DataContainer


def sequential_warmup(model, rounds, train_clients, test_dataset: DataContainer, epochs, lr):
    total_accs = []
    trainer = TorchModel(copy.deepcopy(model))
    last_round_trainers = []
    times = []
    timer = TimeCheckpoint()
    for k in tqdm(range(rounds), desc="Training Rounds"):
        trainer_old: Union[TorchModel, None] = None
        for i, vi in train_clients.items():
            trainer.train(train_clients[i].batch(), lr=lr, epochs=epochs, verbose=0)
            if k != rounds - 1 and trainer_old is not None:
                trainer.dilute(trainer_old, 10)
            trainer_old = trainer.copy()
            if k == rounds - 1:
                last_round_trainers.append(trainer.copy())
        total_accs.append(trainer.infer(test_dataset.batch()))
        times.append(timer.cp())
    models_states = {}
    models_sizes = {}
    for i in range(len(last_round_trainers)):
        models_states[i] = last_round_trainers[i].model.state_dict()
        models_sizes[i] = 1
    weights = federated_tools.aggregate(models_states, models_sizes)
    trainer.load(weights)
    total_accs.append(trainer.infer(test_dataset.batch(), verbose=0))
    logger.info(f"final avg: {total_accs[-1]}")
    return weights, total_accs
