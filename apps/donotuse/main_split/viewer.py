from matplotlib import pyplot as plt

items = {
    'split': [1, 2, 3, 4, 5],
    'split2': [10, 22, 34, 4, 5],
}
plt.grid()
for key, item in items.items():
    p2 = plt.plot(item, '-', label=key, linewidth=5)
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
