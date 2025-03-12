import copy
import typing
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn
import logging
from src.data.data_container import DataContainer

logger = logging.getLogger('tools')


def train(model, train_data, epochs=10, lr=0.1, logging=True):
    torch.cuda.empty_cache()
    # change to train mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    epoch_loss = []
    for epoch in tqdm.tqdm(range(epochs), 'training') if logging else range(epochs):
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(train_data):
            x = x.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            log_probs = model(x)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        if len(batch_loss) > 0:
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    weights = model.state_dict()
    return weights


def aggregate(models_dict: dict, sample_dict: dict):
    model_list = []
    training_num = 0

    for idx in models_dict.keys():
        if idx not in sample_dict:
            sample_dict[idx] = 1
        model_list.append((sample_dict[idx], copy.deepcopy(models_dict[idx])))
        training_num += sample_dict[idx]

    # logging.info("################aggregate: %d" % len(model_list))
    (num0, averaged_params) = model_list[0]
    for k in averaged_params.keys():
        for i in range(0, len(model_list)):
            local_sample_number, local_model_params = model_list[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w

    return averaged_params


def asyncgregate(current_weights, staled_weights, staleness: int):
    """
    Method created to simulate FedAsync, used to dilute the stalled clients weights during the aggregation
    @param current_weights: current model weights
    @param staled_weights: the weights of stalled client having old weights that should be diluted
    @param staleness: should be an int >=1. Equals to the number of round the client missed the training.
    @return:
    """
    alpha = 1. / (1 + staleness)

    for name, param in current_weights.items():
        current_weights[name] = ((1 - alpha) * current_weights[name]) + (alpha * staled_weights[name])
    return current_weights


def infer(model, test_data, transformer=None):
    model.eval()
    test_loss = test_acc = test_total = 0.
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data):
            if transformer:
                x, target = transformer(x, target)
            pred = model(x)
            loss = criterion(pred, target)
            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            test_acc += correct.item()
            test_loss += loss.item() * target.size(0)
            test_total += target.size(0)

    return test_acc / test_total, test_loss / test_total


def infer2(model, batched, **kwargs):
    verbose = kwargs.get('verbose', 1)
    device = kwargs['device'] if 'device' in kwargs else ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    test_loss = test_acc = test_total = 0.
    criterion = nn.CrossEntropyLoss() if 'criterion' not in kwargs else kwargs['criterion']

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        iterator = tqdm.tqdm(enumerate(batched), 'inferring', disable=verbose == 0)
        for batch_idx, (x, target) in iterator:
            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            loss = criterion(pred, target)
            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            test_acc += correct.item()
            test_loss += loss.item() * target.size(0)
            test_total += target.size(0)

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = test_acc / test_total
    avg_loss = test_loss / test_total

    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    return accuracy, avg_loss, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def confusion_matrix(model, batched, **kwargs):
    verbose = kwargs.get('verbose', 1)
    device = kwargs['device'] if 'device' in kwargs else ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    test_loss = test_acc = test_total = 0.
    criterion = nn.CrossEntropyLoss() if 'criterion' not in kwargs else kwargs['criterion']

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        iterator = tqdm.tqdm(enumerate(batched), 'inferring', disable=verbose == 0)
        for batch_idx, (x, target) in iterator:
            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            loss = criterion(pred, target)
            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            test_acc += correct.item()
            test_loss += loss.item() * target.size(0)
            test_total += target.size(0)

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return all_targets, all_predictions


def load(model, stats):
    model.load_state_dict(stats)


def detail(client_data: typing.Union[typing.Dict[int, DataContainer], DataContainer], selection=None,
           display: typing.Callable = None):
    if display is None:
        display = lambda x: logger.info(x)
    if isinstance(client_data, DataContainer):
        client_data = {0: client_data}
    display("<--clients_labels-->")
    for client_id, data in client_data.items():
        if selection is not None:
            if client_id not in selection:
                continue
        uniques = np.unique(data.y)
        display(f"client_id: {client_id} --size: {len(data.y)} --num_labels: {len(uniques)} --unique_labels:{uniques}")
        for unique in uniques:
            unique_count = 0
            for item in data.y:
                if item == unique:
                    unique_count += 1
            percentage = unique_count / len(data.y) * 100
            percentage = int(percentage)
            display(f"labels_{unique}= {percentage}% - {unique_count}")


def plot(tests: typing.Dict[str, typing.List[float]]):
    markers = ['o', 's', 'D', '^', 'v', 'p', '*']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    i = 0
    for name, acc in tests.items():
        plt.plot(range(len(acc)), acc, label=f'Model {name}', marker=markers[i % len(markers)],
                 color=colors[i % len(colors)])
        i += 1

    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()


def compare(m1, m2):
    """

    Args:
        m1: first network
        m2: second network

    Returns: true if same, false otherwise

    """
    state_a = m1.state_dict().__str__()
    state_b = m2.state_dict().__str__()

    return state_a == state_b
