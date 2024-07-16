import pickle
import matplotlib.pyplot as plt
import numpy as np

clients = pickle.load(open('clients.pkl', 'rb'))
data = []
for id, dc in clients.items():
    data.append((len(dc), dc.unique()))

record_sizes = [item[0] for item in data]
label_counts = [len(item[1]) for item in data]

# Create a scatter plot

plt.scatter(label_counts, record_sizes)
plt.xlabel('Number of Unique Labels')
plt.ylabel('Number of Records')
plt.grid(True, linestyle='--', alpha=0.5)

# # Create a histogram
# plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
# plt.hist(label_counts, bins=np.arange(12) - 0.5, edgecolor='black')
# plt.title('Histogram: Distribution of Label Counts')
# plt.xlabel('Number of Unique Labels')
# plt.ylabel('Frequency')

# Show plots
plt.tight_layout()
plt.show()
