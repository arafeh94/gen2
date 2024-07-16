from matplotlib import pyplot as plt

from src.apis import math
from src.apis.rw import IODict

cache = IODict('cache/.tr.iod')
cache.load()
items = cache.all('test_20_5_0.001', 'test_20_5_0.0001')
print(cache.cached.keys())
plt.grid()
for key, item in items.items():
    p2 = plt.plot(item['split'], '-', label=key, linewidth=5)
plt.legend()
plt.show()
