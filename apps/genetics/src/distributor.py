import random

from src.data import data_loader
from src.data.data_distributor import PipeDistributor
from src.data.data_loader import preload


def get_distributor():
    rand = random.randint(0, 10)
    pipes = []
    for i in range(25):
        pipes.append(PipeDistributor.pick_by_label_id(random.sample(range(0, 9), 2), 50))
    for i in range(25):
        pipes.append(PipeDistributor.pick_by_label_id(random.sample(range(0, 9), 5), 50))
    for i in range(25):
        pipes.append(PipeDistributor.pick_by_label_id(random.sample(range(0, 9), 2), 750))
    for i in range(25):
        pipes.append(PipeDistributor.pick_by_label_id(random.sample(range(0, 9), 5), 750))

    return PipeDistributor(pipes)


def get_distributed_data():
    return preload('fall_ar_by_client', get_distributor(), tag='pipe_distributed_data_random_unfair')

