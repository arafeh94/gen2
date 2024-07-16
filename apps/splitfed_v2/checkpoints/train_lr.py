import pickle
import random

import sklearn.metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

from apps.splitfed_v2.core.client import resource_generator, exec_time


def train():
    features = []
    labels = []
    speeds = [0.015, 0.1, 0.5, 1, 1.5, 2, 3, 5]

    for speed in speeds:
        print("started speed: {}".format(speed))

        for epoch in range(10000):
            ram, cpu, disk, latency = resource_generator.generate_one(speed)
            time_taken = exec_time(cpu, ram, disk, latency)
            features.append([ram, cpu, disk, latency])
            labels.append(time_taken)

        print("finished speed: {}".format(speed))

    # Create and train the Linear Regression model
    lr_model = LinearRegression()
    print("fitting Linear Regression model")
    lr_model.fit(features, labels)

    print("writing to file")
    pickle.dump(lr_model, open('../files/lr_model.pkl', 'wb'))


def test():
    model = pickle.load(open('../files/lr_model.pkl', 'rb'))
    speeds = [0.015, 0.1, 0.5, 1, 1.5, 2, 3, 5]
    speeds_res = {}
    for speed in speeds:
        y_true = []
        y_predict = []
        print("started speed: {}".format(speed))
        for epoch in range(200):
            ram, cpu, disk, latency = resource_generator.generate_one(speed)
            time_taken = exec_time(cpu, ram, disk, latency)
            gp_time_taken = model.predict([[ram, cpu, disk, latency]])[0]
            print('true {}, predicted {}'.format(time_taken, gp_time_taken))
            y_true.append(time_taken)
            y_predict.append(gp_time_taken)
        speeds_res[speed] = mean_absolute_error(y_true, y_predict)

    print(speeds_res)


# train()
test()
