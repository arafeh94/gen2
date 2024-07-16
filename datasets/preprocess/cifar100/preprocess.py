import pickle
import sys

import numpy as np

from src.data.data_container import DataContainer


def unpickle(file) -> dict:
    with open(file, 'rb') as fo:
        fo.seek(0)
        dict = pickle.load(fo, encoding='latin1')
    return dict


xx = []
yy = []

xs = []
ys = []
dt = unpickle('./data/train')
data, labels = dt['data'], dt['fine_labels']
for i in range(len(data)):
    xs.append(data[i])
    ys.append(labels[i])
    xx.append(data[i])
    yy.append(labels[i])
train_dc = DataContainer(xs, ys)
pickle.dump(train_dc, open('./cifar100_train.pkl', 'wb'))

xs = []
ys = []
dt = unpickle('./data/test')
data, labels = dt['data'], dt['fine_labels']
for i in range(len(data)):
    xs.append(data[i])
    ys.append(labels[i])
    xx.append(data[i])
    yy.append(labels[i])
test_dc = DataContainer(xs, ys)
pickle.dump(test_dc, open('./cifar100_test.pkl', 'wb'))

all_dc = DataContainer(xx, yy)
pickle.dump(all_dc, open('./cifar100.pkl', 'wb'))
