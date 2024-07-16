import sys

from easydict import EasyDict as edict

from libs.model.linear.lr_kdd import KDD_LR
from libs.model.linear.mnist_net import MnistNet
from src.apis import utils
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload

# distributor = PipeDistributor(
#     [
#         PipeDistributor.pick_by_label_id([1, 2, 4], 1000, 6),
#         PipeDistributor.pick_by_label_id([1, 2, 8], 1000, 6),
#         PipeDistributor.pick_by_label_id([4, 5, 8], 1000, 6),
#         PipeDistributor.pick_by_label_id([5, 6, 7, 8], 4000, 4),
#         PipeDistributor.pick_by_label_id([6, 7], 2000, 5),
#         PipeDistributor.pick_by_label_id([7, 8], 2000, 5),
#         PipeDistributor.pick_by_label_id([0], 200, 15),
#         PipeDistributor.pick_by_label_id([1], 200, 15),
#         PipeDistributor.pick_by_label_id([2], 200, 15),
#         PipeDistributor.pick_by_label_id([3], 200, 15),
#         PipeDistributor.pick_by_label_id([4], 200, 15),
#         PipeDistributor.pick_by_label_id([5], 200, 15),
#         PipeDistributor.pick_by_label_id([6], 200, 15),
#         PipeDistributor.pick_by_label_id([7], 200, 15),
#         PipeDistributor.pick_by_label_id([8], 200, 15),
#         PipeDistributor.pick_by_label_id([9], 200, 15),
#     ], tag='t1'
# )
# distributor = DirichletDistributor(150, 10, 0.1)
# distributor = ShardDistributor(400, 1)

parameters = {
    'selector': [{'rand': ['cr']}, {'ga': [{'buffer': ['proba', 'pop']}, 'cls', 'ppl']}, 'all'],
    'warmup': ['data_ratio', 'epochs', 'lr'],
    'seqop': ['selector_id', 'rounds', 'epochs', 'lr', 'cr'],
    'ewc': ['rounds', 'epochs', 'lr', 'weight', 'selector', 'cr']
}

cr = 10
wlr = 0.01
we = 20
wr = 32
dt = 'kdd'
cls = 10
buffer = 'pop'
exp_tag = '23'
distributor = ShardDistributor(400, 3)
# distributor = DirichletDistributor(100, 21, 0.1)

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
                'data_ratio': 0.1,
                'lr': wlr,
                'epochs': 5,
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


runs = edict(temp([sys.argv[1]]))
