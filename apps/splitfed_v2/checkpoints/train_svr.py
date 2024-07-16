import os
import pickle
import random

import sklearn.metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from apps.splitfed_v2.core.client import resource_generator, exec_time


def train(speeds, tag, count):
    features = []
    labels = []
    scaler = StandardScaler() if not os.path.exists('../files/svr_scaler.pkl') else pickle.load(
        open('../files/svr_scaler.pkl', 'rb'))

    for speed in speeds:
        print("started speed: {}".format(speed))

        for epoch in range(count):
            upto = random.uniform(0.2, 0.8)
            ram, cpu, disk, _ = resource_generator.generate_one(speed * upto)
            ram, cpu, disk, = round(ram, 1), round(cpu, 1), round(disk, 1)
            if ram == 0 or cpu == 0:
                continue
            time_taken = int(exec_time(cpu, ram, disk))
            features.append([ram, cpu, disk])
            labels.append(time_taken)

        print("finished speed: {}".format(speed))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

    # Standardize the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the Support Vector Regression model
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    print("fitting SVR model")
    svr_model.fit(X_train_scaled, y_train)

    print("writing to file")
    pickle.dump(svr_model, open('../files/{}_svr_model.pkl'.format(tag), 'wb'))
    pickle.dump(scaler, open('../files/svr_scaler.pkl', 'wb'))


def test(speeds, tag):
    # Load the trained SVR model from the file
    model = pickle.load(open('../files/{}_svr_model.pkl'.format(tag), 'rb'))
    scaler = pickle.load(open('../files/svr_scaler.pkl', 'rb'))

    speeds_res = {}

    for speed in speeds:
        y_true = []
        y_predict = []

        print("started speed: {}".format(speed))

        for epoch in range(200):
            upto = random.uniform(0.2, 0.8)
            ram, cpu, disk, _ = resource_generator.generate_one(speed * 0.2)
            ram, cpu, disk, = round(ram, 1), round(cpu, 1), round(disk, 1)
            if ram == 0 or cpu == 0:
                continue
            time_taken = int(exec_time(cpu, ram, disk))

            new_data_scaled = scaler.transform([[ram, cpu, disk]])

            svr_time_taken = model.predict(new_data_scaled)[0]

            print('true {}, predicted {}'.format(time_taken, svr_time_taken))

            y_true.append(time_taken)
            y_predict.append(svr_time_taken)

        speeds_res[speed] = mean_absolute_error(y_true, y_predict)

    print(speeds_res)


train([0.1, 0.25, 1], 't1', 10000)
test([0.1, 0.25, 1], 't1')

# train([0.01, 0.025, 0.05], 'elow', 10_000)
# test([0.01, 0.025, 0.05], 'elow')

# train([0.1, 0.25, 0.5], 'low', 10_000)
# test([0.1, 0.25, 0.5], 'low')

# train([1, 1.5, 2, 2.5], 'medium', 10_000)
# test([1, 1.5, 2, 2.5], 'medium')

# train([3, 4, 5], 'high', 10_000)
# test([3, 4, 5], 'high')
