import random

import numpy as np

from apps.splitfed_v2.core.client import resource_generator, exec_time


def generate_data(speeds, count_each):
    features = []
    labels = []

    for speed in speeds:
        for epoch in range(count_each):
            upto = random.uniform(0.2, 0.8)
            ram, cpu, disk, _ = resource_generator.generate_one(speed * upto)
            ram, cpu, disk, = round(ram, 1), round(cpu, 1), round(disk, 1)
            if ram == 0 or cpu == 0:
                continue
            time_taken = int(exec_time(cpu, ram, disk))
            features.append([ram, cpu, disk])
            labels.append(time_taken)
    return np.array(features), np.array(labels)
