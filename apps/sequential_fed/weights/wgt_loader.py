import pickle

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

from libs.model.linear.mnist_net import MnistNet
from src.apis import federated_tools
from src.data.data_loader import preload

file = open('t1723442391.pkl', 'rb')
weights = pickle.load(file)
model = MnistNet(28 * 28, 32, 10)
model.load_state_dict(weights)
test = preload('mnist10k').as_tensor()
all_targets, all_predictions = federated_tools.confusion_matrix(model, test.batch())
print("targets", all_targets)
print("predictions", all_predictions)
cm = confusion_matrix(all_targets, all_predictions, labels=np.arange(10))
print(cm)
# Plot confusion matrix using only Matplotlib
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(cm, cmap='Blues')

# Add color bar
plt.colorbar(cax)

# Add labels
classes = np.arange(10)
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# Add text annotations
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
