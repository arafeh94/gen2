import numpy as np
from sklearn.decomposition import PCA

from apps.splitfed.models import MnistNet
from src.apis import utils
from src.apis.extensions import TorchModel
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import mnist


def test(pca, data):
    data_transformed = pca.transform(data)
    data_reconstructed = pca.inverse_transform(data_transformed)
    mse = np.mean((data - data_reconstructed) ** 2)
    return mse


distributor = ShardDistributor(400, 1)
dd2 = ShardDistributor(100, 5)
d1 = distributor.distribute(mnist().filter(lambda x, y: y == 1))
d2 = distributor.distribute(mnist().filter(lambda x, y: y == 2))
d3 = dd2.distribute(mnist())
trainers_d1 = []
trainers_d2 = []
trainers_d3 = []

for i in range(10):
    trainer_d1 = TorchModel(MnistNet(28 * 28, 128, 10))
    trainer_d2 = TorchModel(MnistNet(28 * 28, 128, 10))
    trainer_d3 = TorchModel(MnistNet(28 * 28, 128, 10))
    trainer_d1.train(d1[i].batch(), epochs=10)
    trainer_d2.train(d2[i].batch(), epochs=10)
    trainer_d3.train(d3[i].batch(), epochs=10)
    trainers_d1.append(trainer_d1)
    trainers_d2.append(trainer_d2)
    trainers_d3.append(trainer_d3)

x1 = [utils.flatten_weights(t.state()) for t in trainers_d1]
x2 = [utils.flatten_weights(t.state()) for t in trainers_d2]
x3 = [utils.flatten_weights(t.state()) for t in trainers_d3]

x1_transformed = PCA().fit_transform(x1)
print(x1_transformed)
