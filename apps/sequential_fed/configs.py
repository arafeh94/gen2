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
we = 2
wr = 2
dt = 'mnist'
cls = 10
buffer = 'pop'
exp_tag = 'add01'
# distributor = ShardDistributor(400, 1)
distributor = DirichletDistributor(100, 21, 0.1)

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


def temp(items):
    return utils.dict_select(items, {
        'warmup': {
            'id': f'warmup_{dt}_all_{500}_{500}_{wlr}',
            'method': 'warmup',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'data_ratio': 0.05,
                'lr': 0.01,
                'epochs': 500,
            },
            'fed': fed_config,
        },
        'seq1': {
            'id': f'seqop_{dt}_all_{wr}_{we}_{wlr}',
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
        'seq2': {
            'id': f'seqop_{dt}_ga{cr}_{wr}_{we}_{wlr}',
            'method': 'seqop_ga',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'ga',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'cr': cr,
                'buffer': 'pop',
                'cls': cls,
            },
            'fed': fed_config,
        },
        'seq3': {
            'id': f'seqop_{dt}_rn{cr}_{wr}_{we}_{wlr}',
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
                'cls': cls,
            },
            'fed': fed_config,
        },
        'ewc1': {
            'id': f'ewc_{dt}_all_{wr}_{we}_{wlr}',
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
        'ewc2': {
            'id': f'ewc_{dt}_ga{cr}_{wr}_{we}_{wlr}',
            'method': 'ewc_ga',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'ga',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'weight': 0.1,
                'cr': cr,
                'buffer': 'pop',
                'cls': cls,
            },
            'fed': fed_config,
        },
        'ewc3': {
            'id': f'ewc_{dt}_rn{cr}_{wr}_{we}_{wlr}',
            'method': 'ewc_rn',
            'distributor': str(distributor),
            'tag': exp_tag,
            'wmp': {
                'selector': 'rn',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'weight': 0.1,
                'cr': cr,
            },
            'fed': fed_config,
        },
    }
                             )


runs = edict(temp([sys.argv[1] if len(sys.argv) > 1 else 'warmup']))
