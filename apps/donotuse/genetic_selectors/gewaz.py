import logging

from apps.donotuse.genetic_selectors import distributor
from apps.donotuse.genetic_selectors.algo import initializer

from libs.model.linear.lr import LogisticRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

model = lambda: LogisticRegression(28 * 28, 10)
db_name = 'res7.db'
logger.info('Generating Data --Started')
client_data = distributor.get_distributed_data()

# 100
# 10 - c_size
# 200*10 - p_size = total number of chromosomes


resources = {}
initial_model = initializer.ga_resource_selector(resources, max_iter=200,
                                                 r_cross=0.5, r_mut=0.1,
                                                 c_size=20, p_size=200,
                                                 desired_fitness=0.9)
