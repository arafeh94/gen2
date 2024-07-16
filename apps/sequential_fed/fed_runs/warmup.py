import pickle
import typing

from apps.sequential_fed.fed_runs.seq_ga import CNN
from apps.sequential_fed.s_core import pfed
from libs.model.cv.cnn import Cifar10Model
from src.apis import lambdas, utils
from src.apis.extensions import TorchModel
from src.data.data_container import DataContainer
from src.federated.subscribers.sqlite_logger import SQLiteLogger

utils.enable_logging()

clients = pickle.load(open('clients.pkl', 'rb'))

clients = clients.map(lambdas.as_numpy)
shared = clients.map(lambda cid, dc: dc.split(0.05)[0]).reduce(lambdas.dict2dc)
clients = clients.map(lambda cid, dc: dc.split(0.05)[1]).map(lambda cid, dc: dc.concat(shared)).map(lambdas.as_tensor)
test = pickle.load(open('test.pkl', 'rb'))
model = CNN()
db_logger = SQLiteLogger.new_instance('seqfed2.sqlite', {'method': 'warmup_initials'})
#
# trainer = TorchModel(model)
# weights, accs = trainer.train(shared.as_tensor().batch(150), lr=0.001, epochs=20_000)
# pickle.dump(weights, open('warmup_initial_weights.pkl', 'wb'))
# [db_logger.log(i, acc=acc) for i, acc in enumerate(accs)]

#
# fed_config = {'rounds': 10000, 'lr': 0.001, 'epochs': 20, 'cr': 10}
# initial_weights = pickle.load(open('warmup_initial_weights.pkl', 'rb'))
# model.load_state_dict(initial_weights)
# federate = pfed.create_fl(clients, test, model, fed_config, db_logger)
# federate.start()
