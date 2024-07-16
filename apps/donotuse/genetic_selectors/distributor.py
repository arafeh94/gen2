import random

from src.data import data_loader
from src.data.data_distributor import PipeDistributor
from src.data.data_loader import preload


def get_distributor():
    rand = random.randint(0, 10)
    pipes = []
    for i in range(25):
        pipes.append(PipeDistributor.pick_by_label_id(random.sample(range(0, 9), 4), 20))
    for i in range(75):
        pipes.append(PipeDistributor.pick_by_label_id(random.sample(range(0, 9), 4), 500))

    return PipeDistributor(pipes)


def get_distributed_data():
    return preload('mnist', get_distributor(), tag='pipe_distributed_data_random_2')
