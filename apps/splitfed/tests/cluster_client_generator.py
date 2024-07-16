from apps.splitfed.core.clusters import generate_resources
from libs.model.linear.lr import LogisticRegression
from src.apis import utils
from src.data.data_distributor import ClusterDistributor, ShardDistributor
from src.data.data_loader import preload

utils.enable_logging()
# this will create 5 cluster (clients), 2 label each
inner_distributor = ShardDistributor(10000, 2)
# fill client in each cluster from mnist dataset, each cluster will have 10 clients of 200 records each
clients_clusters = preload('mnist', ClusterDistributor(inner_distributor, 10, 200, 200))
# separate the client in each cluster by speed
clusters = generate_resources(clients_clusters, LogisticRegression(28 * 28, 10), 5, [0.5, 1, 2])
print(clusters)
