import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from apps.splitfed_v2.checkpoints.train_generate import generate_data
from apps.splitfed_v2.core.models import RegressionModel


# Define the neural network model


def train(data_size, num_epochs):
    X, y = generate_data([0.1, 0.25, 1], data_size)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    X = torch.tensor(X).float().to('cuda')
    y = torch.tensor(y).float().to('cuda')
    model = RegressionModel()
    model.to('cuda')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        predictions = model(X)
        loss = criterion(predictions, y.view(-1, 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pickle.dump(model, open('../files/nn_model.pkl', 'wb'))


def test():
    model = pickle.load(open('../files/nn_model.pkl', 'rb'))
    test_X, test_y = generate_data([0.1, 0.25, 1], 300)
    test_X = torch.tensor(test_X).float().to('cuda')
    test_y = torch.tensor(test_y).float().to('cuda')

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        test_predictions = model(test_X)

    # Convert predictions to numpy array
    test_predictions = test_predictions.cpu().numpy()
    test_y = test_y.cpu().numpy()
    mse = mean_squared_error(test_y, test_predictions)
    print(mse)


train(10000, 40000)
test()
