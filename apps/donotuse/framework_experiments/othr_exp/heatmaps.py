from matplotlib import pyplot as plt

from src.data.data_distributor import UniqueDistributor, LabelDistributor, DirichletDistributor, ShardDistributor
from src.data.data_loader import preload
from src.federated.subscribers.analysis import ShowDataDistribution

shards = '##1'
labels = '##2'
skewness = '##3'
records_per_clients = '##4'
shard_size = '##5'

# ShowDataDistribution.plot(preload('mnist', ShardDistributor(300, 2)).select(range(20)), 10, title='',
#                           sub_title='')
#
# ShowDataDistribution.plot(preload('mnist', UniqueDistributor(10, 3000, 3000)), 10, title='',
#                           sub_title='')
#
# ShowDataDistribution.plot(preload('mnist', LabelDistributor(20, 10, 600, 600)), 10, title='',
#                           sub_title='',
#                           text_color='white')
#
# ShowDataDistribution.plot(preload('mnist', LabelDistributor(20, 1, 600, 600)), 10, title='',
#                           sub_title='')

ShowDataDistribution.plot(preload('mnist', DirichletDistributor(20, 10, 0.1)), 10, title='',
                          sub_title='')
#
# ShowDataDistribution.plot(preload('mnist', DirichletDistributor(20, 10, 0.5)), 10, title='',
#                           sub_title='', text_color='white')

# ShowDataDistribution.plot(preload('mnist', UniqueDistributor(10, 3000, 3000)), 10, title='Unique Distribution')
#
# ShowDataDistribution.plot(preload('mnist', LabelDistributor(20, 10, 600, 600)), 10, title='Label Distribution')
# ShowDataDistribution.plot(preload('mnist', LabelDistributor(20, 3, 600, 600)), 10, title='Label Distribution')
# ShowDataDistribution.plot(preload('mnist', LabelDistributor(20, 1, 600, 600)), 10, title='Label Distribution')

# ShowDataDistribution.plot(preload('mnist', DirichletDistributor(20, 10, 10)), 10, title='Unique Distribution')
# ShowDataDistribution.plot(preload('mnist', DirichletDistributor(20, 10, 0.5)), 10, title='Unique Distribution')
