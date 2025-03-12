import sys

from easydict import EasyDict as edict

from libs.model.linear.lr_kdd import KDD_LR
from libs.model.linear.mnist_net import MnistNet
from src.apis import utils
from src.data.data_distributor import ShardDistributor, DirichletDistributor
from src.data.data_loader import preload

parameters = {
    'selector': [{'rand': ['cr']}, {'ga': [{'buffer': ['proba', 'pop']}, 'cls', 'ppl']}, 'all'],
    'warmup': ['data_ratio', 'epochs', 'lr'],
    'seqop': ['selector_id', 'rounds', 'epochs', 'lr', 'cr'],
    'ewc': ['rounds', 'epochs', 'lr', 'weight', 'selector', 'cr']
}

cr = 10
wlr = 0.01
we = 20
wr = 20
dt = 'mnist'
cls = 10

mut = 0.05
cross = 0.1
p_size = 50
max_iter = 50

buffer = 'pop'
exp_tag = 'add01'
# distributor = ShardDistributor(400, 1)
distributor = DirichletDistributor(150, 10, 0.1)

if dt == 'kdd':
    train, test = preload("fekdd_train").filter(lambda x, y: y not in [21, 22, 23]).split(0.8)
    test = test.as_tensor()
    base_model = KDD_LR(41, 23)

if dt == 'mnist':
    train, test = preload("mnist").split(0.8)
    test = test.as_tensor()
    base_model = MnistNet(28 * 28, 32, 10)

fed_config = {
    'rounds': 300,
    'lr': 0.01,
    'epochs': 25,
    'cr': 10
}


def wrapper(items):
    all_configs = {
        'warmup1': {
            'method': 'warmup',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'data_ratio': 0.05,
                'lr': 0.01,
                'epochs': 1000,
            },
            'fed': fed_config,
        },
        'warmup2': {
            'method': 'warmup',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'data_ratio': 0.1,
                'lr': 0.01,
                'epochs': 1000,
            },
            'fed': fed_config,
        },
        'seq11': {
            'method': 'seqop_all',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'all',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'buffer': 'pop',
                'cls': cls,
            },
            'fed': fed_config,
        },
        'seq21': {
            'method': 'seqop_ga',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'ga',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'cr': 10,
                'buffer': 'pop',
                'cls': 5,
                'mut': 0.05,
                'cross': 0.1,
                'p_size': 50,
                'max_iter': 50,
            },
            'fed': fed_config,
        },
        'seq22': {
            'method': 'seqop_ga',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'ga',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'cr': 10,
                'buffer': 'pop',
                'cls': 5,
                'mut': 0.1,
                'cross': 0.3,
                'p_size': 100,
                'max_iter': 100,
            },
            'fed': fed_config,
        },
        'seq23': {
            'method': 'seqop_ga',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'ga',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'cr': 10,
                'buffer': 'pop',
                'cls': 10,
                'mut': 0.05,
                'cross': 0.1,
                'p_size': 50,
                'max_iter': 50,
            },
            'fed': fed_config,
        },
        'seq24': {
            'method': 'seqop_ga',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'ga',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'cr': 10,
                'buffer': 'pop',
                'cls': 10,
                'mut': 0.1,
                'cross': 0.3,
                'p_size': 100,
                'max_iter': 100,
            },
            'fed': fed_config,
        },
        'seq31': {
            'method': 'seqop_rn',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'rn',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'cr': cr,
                'buffer': 'pop',
            },
            'fed': fed_config,
        },
        'seq32': {
            'method': 'seqop_rn',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'rn',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'cr': 5,
                'buffer': 'pop',
            },
            'fed': fed_config,
        },
        'ewc11': {
            'method': 'ewc_all',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'all',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'weight': 0.1,
            },
            'fed': fed_config,
        },
        'ewc12': {
            'method': 'ewc_all',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'all',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'weight': 0.5,
            },
            'fed': fed_config,
        },
        'ewc31': {
            'method': 'ewc_rn',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'rn',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'weight': 0.1,
                'cr': 5,
            },
            'fed': fed_config,
        },
        'ewc32': {
            'method': 'ewc_rn',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'rn',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'weight': 0.1,
                'cr': 10,
            },
            'fed': fed_config,
        },
    }
    return utils.dict_select(items, all_configs)


run_ids = wrapper([sys.argv[1] if len(sys.argv) > 1 else 'warmup'])
runs = edict(run_ids)
