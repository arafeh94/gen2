from matplotlib import pyplot as plt

from src.apis.rw import IODict

results = IODict('./res')
plt.grid()
plt.plot(results['normal_fedsplit'], '-', label='SplitFed-Normal', linewidth=5)
plt.plot(results['cluster_fedsplit'], '-', label='SplitFed-Clustered', linewidth=5)
plt.plot(results['original_split'], '-', label='Split-Original', linewidth=5)
plt.legend()
plt.show()
