from src.apis import utils
from src.data.data_loader import preload

utils.enable_logging()
dt = preload('cifar100_train')
print(dt)
