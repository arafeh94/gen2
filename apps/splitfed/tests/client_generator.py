import copy

from apps.splitfed.core.client import Client
from libs.model.linear.lr import LogisticRegression
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload

model = LogisticRegression(28 * 28, 10)
clients_data = preload('mnist', ShardDistributor(300, 3))
client = Client.generate(clients_data[0], model, 1)
client.randomize_resources()
print(client.configs)
print(client.exec_time())
client.randomize_resources(0.1)
print(client.configs)
print(client.exec_time())
client.randomize_resources(0.5)
print(client.configs)
print(client.exec_time())
client.randomize_resources(0.8)
print(client.configs)
print(client.exec_time())
