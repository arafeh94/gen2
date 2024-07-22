import json
import logging
import subprocess
import sys


sys.path.append('../../')

from apps.donotuse.split_learning import models
from src.apis import utils, lambdas
from src.apis.extensions import Dict
from src.data.data_distributor import ShardDistributor, DirichletDistributor
from src.data.data_loader import preload

# Important Nodes:
# (!1) for cifar with higher parameters:
#       -change model names (client, server, model)
#       -change dataset loader dt_tag name (use cifar10_dir2 instead of dir1)
# (!2) change the tag name in __name__ before doing experiment to makes it easier to get results correct
# (!3) run federated alone separately


dataset_name = 'cifar10'
if dataset_name == 'cifar10':
    def loader(t_dataset):
        t_dataset = t_dataset.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
        i_train, i_test = t_dataset.shuffle(27).split(0.9)
        i_train = DirichletDistributor(150, 10, 0.1).distribute(i_train).map(lambdas.mappers.as_tensor)
        i_test = i_test.as_tensor()
        return i_train, i_test


    train, test = preload('cifar10', tag='cifar10_train_test_2', transformer=loader)

else:
    train = preload('mnist', DirichletDistributor(150, 10, 0.1))
    test = preload('mnist10k').as_tensor()

global_configs = Dict({
    'rounds': 5000,
    'lr_client': 0.001,
    'lr_server': 0.001,
    'batch': 0,
    'epoch': 1,
    # 'client_model': models.MnistClient(784, 1024, 10),
    # 'server_model': models.MnistServer(784, 1024, 10),
    # 'model': models.MnistNet(784, 1024, 10),
    'client_model': models.CifarClient2(),
    'server_model': models.CifarServer2(),
    'model': models.CifarModel2(),
    'train': train,
    'test': test,
    'cls_speeds': [.1, .25, 1],
    'out_size': 3,
    'bad_ratio': .05,
    'dt_tag': f'{dataset_name}_dir2',
})

if __name__ == '__main__':
    runs = [
        # './sfed.py',
        # './split.py',
        # './splitfed.py',
        './splitfed1layer.py',
        # './splitfed2layers_selection.py',
        # './splitfed2layers_standard.py',
    ]
    logger = logging.getLogger('_run')
    for path in runs:
        logger.error('--------------Starting {} Execution--------------'.format(path))
        track_params = ['rounds', 'lr_client', 'lr_server', 'cls_speeds', 'out_size', 'bad_ratio', 'dt_tag']
        configs = json.dumps({'name': path, 'tag': 'cifar_temp', **global_configs.select(track_params)})
        subprocess.run([utils.venv(), path, configs])
        logger.error('--------------{} Finished Execution--------------'.format(path))
