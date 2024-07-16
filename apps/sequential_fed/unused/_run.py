from apps.sequential_fed.configs import fed_config
from apps.sequential_fed.main_def import run
from src.data.data_distributor import ShardDistributor, DirichletDistributor, PipeDistributor
from easydict import EasyDict as edict
from itertools import product

dirichlet = DirichletDistributor(120, 26, 0.1)
shard_no_iid = ShardDistributor(400, 1)
shard_iid = ShardDistributor(100, 5)

configs_parameters = {
    'cr': [5, 10, 20],
    'wlr': [0.1, 0.01, 0.001, 0.001],
    'we': [1, 5, 10, 20, 25],
    'wr': [10, 32, 50, 100],
    'dist': [dirichlet, shard_no_iid, shard_iid]
}


def generate_combinations(dict_params):
    keys = dict_params.keys()
    values = dict_params.values()
    all_combinations = list(product(*values))
    result_combinations = [dict(zip(keys, combination)) for combination in all_combinations]
    return result_combinations


def configs_template(dt, wlr, cr, distributor, wr, we):
    return edict({
        'warmup': {
            'id': f'warmup_{dt}_all_{500}_{500}_{wlr}',
            'method': 'warmup',
            'distributor': distributor,
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
            'distributor': distributor,
            'wmp': {
                'selector': 'all',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
            },
            'fed': fed_config,
        },
        'seq2': {
            'id': f'seqop_{dt}_ga{cr}_{wr}_{we}_{wlr}',
            'method': 'seqop_ga',
            'distributor': distributor,
            'wmp': {
                'selector': 'ga',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'cr': cr,
            },
            'fed': fed_config,
        },
        'seq3': {
            'id': f'seqop_{dt}_rn{cr}_{wr}_{we}_{wlr}',
            'method': 'seqop_rn',
            'distributor': distributor,
            'wmp': {
                'selector': 'rn',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'cr': cr,
            },
            'fed': fed_config,
        },
        'ewc1': {
            'id': f'ewc_{dt}_all_{wr}_{we}_{wlr}',
            'method': 'ewc_all',
            'distributor': distributor,
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
            'distributor': distributor,
            'wmp': {
                'selector': 'ga',
                'rounds': wr,
                'epochs': we,
                'lr': wlr,
                'weight': 0.1,
                'cr': cr,
            },
            'fed': fed_config,
        },
        'ewc3': {
            'id': f'ewc_{dt}_rn{cr}_{wr}_{we}_{wlr}',
            'method': 'ewc_rn',
            'distributor': distributor,
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
    })


executions = []
generated_configs = generate_combinations(configs_parameters)
for gc in generated_configs:
    configs = configs_template("mnist", gc['wlr'], gc['cr'], gc['dist'], gc['wr'], gc['we'])
    executions.append(configs)

for execution in executions:
    run(execution, 'mnist')
