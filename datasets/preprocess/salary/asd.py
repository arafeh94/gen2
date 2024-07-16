import pickle

from src.data.data_loader import preload

dataset = preload('salary').as_list()
print(dataset)