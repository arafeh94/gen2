import multiprocessing
import os
import sys

from src.apis.fed_sqlite import FedDB

sys.path.append("../../../")
from src.apis import test_cases


def build_tag(args):
    return f'{args["tag"]}_e{args["epoch"]}_b{args["batch"]}_r{args["round"]}_dis#{args["distributor"]}#' \
           f'_{args["dataset"]}_cr{str(args["client_ratio"]).replace(".", "")}' \
           f'_lr{str(args["learn_rate"])}'.replace('cr1', 'cr10')


def run_command(execution, force=False):
    save_tag = build_tag(execution)
    if force or save_tag not in FedDB('res.db').tables().values():
        print(f'Executing: {save_tag}')
        os.system(
            f"py {execution['script']}.py  -e {execution['epoch']} -b {execution['batch']} -r {execution['round']} "
            f"-s {execution['distributor']} -d {execution['dataset']} -cr {execution['client_ratio']} -lr {execution['learn_rate']} "
            f"-t {execution['tag']}")
    else:
        print("Experiments already executed, skip.")


key_tests = {
    # 'mnist_log': test_cases.build({
    #     'epoch': [1],
    #     'batch': [25],
    #     'round': [100],
    #     'client_ratio': [10],
    #     'dataset': ['mnist'],
    #     'learn_rate': [0.1],
    #     'script': ['exp_mnist'],
    #     'tag': ['mnist_selector_random_2'],
    #     'distributor': ['mnist_dir_05']
    # }),
    'mnist_wd_1': test_cases.build({
        'epoch': [1],
        'batch': [25],
        'round': [100],
        'client_ratio': [10],
        'dataset': ['mnist'],
        'learn_rate': [0.1],
        'script': ['exp_mnist_wd'],
        'tag': ['wd1'],
        'distributor': ['mnist_c10_label_1']
    }),
    'mnist_wd_2': test_cases.build({
        'epoch': [1],
        'batch': [25],
        'round': [100],
        'client_ratio': [10],
        'dataset': ['mnist'],
        'learn_rate': [0.1],
        'script': ['exp_mnist_wd'],
        'tag': ['wd2'],
        'distributor': ['mnist_c10_label_10']
    }),
}

if __name__ == '__main__':
    # pool = multiprocessing.Pool(2)
    # pool.map(run_command, all_tests)
    for kt in key_tests:
        print(f"starting {kt} tests: {len(key_tests[kt])}")
        for t in key_tests[kt]:
            run_command(t, True)
