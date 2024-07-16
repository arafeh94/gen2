import csv
import logging

import h5py

from src.data.data_container import DataContainer
from src.data.data_provider import PickleDataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

path_train = "heart_cleveland_data.csv"

x = []
y = []
with open(path_train, 'r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)
    for row in csv_reader:
        x.append([float(r) for r in row[:-1]])
        y.append(int(row[-1]))

dc = DataContainer(x, y)
print(dc)
print("saving...")
PickleDataProvider.save(dc, 'heart.pkl')
