import pickle
import matplotlib.pyplot as plt

optimized = pickle.load(open("selector2.pkl", "rb"))
normal = pickle.load(open("no_selector2.pkl", "rb"))


def collect(speed, history, attribute):
    res = []
    for key, round_stats in history.items():
        for stats in round_stats:
            if stats["speed"] == speed:
                res.append(stats[attribute])
    return res


def show_c(cluster_id):
    y1 = collect(cluster_id, optimized, 'tt')
    y2 = collect(cluster_id, normal, 'tt')

    plt.plot(list(range(1, len(y1) + 1)), y1, label='Optimized', color='blue', marker='s')
    plt.plot(list(range(1, len(y2) + 1)), y2, label='Normal', color='red', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Ex. Time')
    plt.title(
        'Cluster({}) - Compare: Optimized ({}) , Normal ({})'
        .format(cluster_id, round(sum(y1) / 1000), round(sum(y2) / 1000))
    )
    plt.legend()
    plt.show()


for i in [.1, .25, 1]:
    show_c(i)
