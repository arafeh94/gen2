import matplotlib.pyplot as plt

from apps.donotuse.framework_experiments import tools

plt.rcParams.update({'font.size': 28})

queries = [
    ['femnist_shard_new', 'r500'],
    ['femnist_dir_new', 'r500'],
    ['cifar_dir_new', 'r500'],
    ['cifar_shard_new', 'r500'],
    ['mnist', 'logistic', 'dir', 'r1000'],
    ['mnist', 'logistic', 'shard', 'r1000'],
    ['mnist', 'logistic', 'unique', 'r1000'],
    ['mnist', 'logistic_e', 'r1000'],

]

for query in queries:
    tools.pretty(tools.filter(*query))
    plts_configs = tools.filter(*query)
    tools.preprocess(plts_configs)
    file_name = "-".join(query)
    tools.plot(tools.plot_builder('acc', plts_configs), f'{file_name}.png', False)
