from src.apis import test_cases, files

files.accuracies.get_saved_accuracy()
exit()

all_tests = test_cases.build({
    'epoch': [25],
    'batch': [50],
    'round': [200],
    'client_ratio': [0.1, 0.2],
    'dataset': ['cifar10'],
    'learn_rate': [0.001, 0.01],
    'tag': ['basic', 'warmup', 'genetic', 'cluster'],
    # 'tag': ['genetic'],
    'shard': [2, 4]
})


def build_tag(args):
    return f'{args["tag"]}_e{args["epoch"]}_b{args["batch"]}_r{args["round"]}_s{args["shard"]}' \
           f'_{args["dataset"]}_cr{str(args["client_ratio"]).replace(".", "")}' \
           f'_lr{str(args["learn_rate"])}'.replace('cr1', 'cr10')


uncomplete = []

for test in all_tests:
    t = build_tag(test)
    if t not in files.accuracies.get_saved_accuracy():
        uncomplete.append(t)

for tt in uncomplete:
    print(tt)
