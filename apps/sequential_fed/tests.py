from torch import nn
from tqdm import tqdm

from apps.sequential_fed.s_core import warmups
from libs.model.linear.lr_kdd import KDD_LR
from libs.model.linear.mnist_net import MnistNet
from src.apis import federated_tools, lambdas, utils
from src.apis.ewc import ElasticWeightConsolidation
from src.apis.utils import TimeCheckpoint
from src.data.data_distributor import ShardDistributor, DirichletDistributor
from src.data.data_loader import preload
from easydict import EasyDict as edict

distributor = DirichletDistributor(120, 26, 0.1)

# train, test = preload("mnist").as_tensor().split(0.8)
# train_clients = distributor.distribute(train).map(lambdas.as_tensor)
# base_model = MnistNet(28 * 28, 32, 10)

train, test = preload("fekdd_train").filter(lambda x, y: y not in [21, 22, 23]).as_tensor().split(0.8)
train_clients = distributor.distribute(train).map(lambdas.as_tensor)
base_model = KDD_LR(41, 23)

# configs
method = 'seq'
rounds = 20
cr = 20
lr = 0.01
acc = []

if method == 'ewc':
    ewc = ElasticWeightConsolidation(base_model, nn.CrossEntropyLoss(), lr=lr, weight=.001)
    for _ in tqdm(range(rounds), desc="Training Rounds"):
        clients = train_clients.select(range(cr))
        for cid, dt in clients.items():
            ewc.train(dt, epochs=1)
        acc_loss = federated_tools.infer(ewc.model, test.batch())
        acc.append(acc_loss[0])
elif method == 'seq':
    time = TimeCheckpoint()
    clients = train_clients
    _, total_accs, _, selector = warmups.sequential_warmup_op(base_model, rounds, clients, test, 5, lr, 'all',
                                                              edict({'wmp': {'cr': cr, 'buffer': 'pop', 'cls': 15}}))
    # utils.plot_bar_dictionary(selector.selection_history)
    acc = [(t[0], t[1]) for t in total_accs]
    print(time.checkpoint())
else:
    for _ in tqdm(range(rounds), desc="Training Rounds"):
        clients = train_clients.select(range(cr))
        for cid, dt in clients.items():
            federated_tools.train(base_model, dt.batch(), epochs=1, logging=False, lr=0.001)
        acc_loss = federated_tools.infer(base_model, test.batch())
        acc.append(acc_loss[0])

print(acc)
print(acc[-1])
