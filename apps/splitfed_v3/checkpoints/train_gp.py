import pickle
import random

import sklearn.metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error

from apps.splitfed_v2.core.client import resource_generator, exec_time


def train():
    features = []
    labels = []
    speeds = [0.01, 0.025, 0.05]
    for speed in speeds:
        print("started speed: {}".format(speed))
        for epoch in range(200):
            ram, cpu, disk, latency = resource_generator.generate_one(speed)
            time_taken = exec_time(cpu, ram, disk, latency)
            features.append([ram, cpu, disk, latency])
            labels.append(time_taken)
        print("finished speed: {}".format(speed))
    kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * 4, (1e-2, 1e2))
    # kernel = 1.0 * RBF(length_scale=1.0)
    gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    print("fitting gaussian process")
    gp_model.fit(features, labels)
    print("writing to file")
    pickle.dump(gp_model, open('../files/elow_gp_model.pkl', 'wb'))


def test():
    gp_model: GaussianProcessRegressor = pickle.load(open('../files/elow_gp_model.pkl', 'rb'))
    speeds = [0.01, 0.025, 0.05]
    speeds_res = {}
    for speed in speeds:
        y_true = []
        y_predict = []
        print("started speed: {}".format(speed))
        for epoch in range(200):
            ram, cpu, disk, latency = resource_generator.generate_one(speed)
            time_taken = exec_time(cpu, ram, disk, latency)
            gp_time_taken = gp_model.predict([[ram, cpu, disk, latency]])[0]
            print('true {}, predicted {}'.format(time_taken, gp_time_taken))
            y_true.append(time_taken)
            y_predict.append(gp_time_taken)
        speeds_res[speed] = mean_absolute_error(y_true, y_predict)

    print(speeds_res)


train()
test()
