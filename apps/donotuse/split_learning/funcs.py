import copy
import os
import pickle

import torch
from torch import nn

from src.data.data_loader import preload


def as_dict(items: list):
    d = {}
    for index, item in enumerate(items):
        d[index] = item
    return d


def average_weights(w, datasize):
    """
    Returns the average of the weights.
    """

    for i, data in enumerate(datasize):
        for key in w[i].keys():
            w[i][key] *= float(data)

    w_avg = copy.deepcopy(w[0])

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

    return w_avg


def get_test_data():
    return preload('mnist10k').as_tensor()


def infer(server_model, client_model, data):
    client_model.eval()
    server_model.eval()
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        corr_num = 0
        total_num = 0
        val_loss = 0.0
        val_x, val_label = data.x, data.y
        val_x = val_x.to(device)
        val_label = val_label.clone().detach().long().to(device)

        val_output = client_model(val_x)
        val_output = server_model(val_output)
        loss = criterion(val_output, val_label)
        val_loss += loss.item()
        model_label = val_output.argmax(dim=1)
        corr = val_label[val_label == model_label].size(0)
        corr_num += corr
        total_num += val_label.size(0)
        test_accuracy = corr_num / total_num
        return test_accuracy, val_loss



