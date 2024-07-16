import random
from random import randint

from matplotlib import pyplot as plt, patches


random.seed(10)
x = []
y = []
size = 100
min = -10
max = 10
subset_size = 4
fig = plt.figure()
ax = fig.add_subplot(111)
latest_lines = []
pause_sec = 0.2


def pause(sec=None):
    plt.pause(sec or pause_sec)


def generate_points():
    while len(x) < size:
        x.append(randint(min, max))
        y.append(randint(min, max))


def plot_lines(sx, sy):
    colors = ["#8e9b90", "#93c0a4", "#b6c4a2", "#d4cdab", "#dce2bd"]
    global latest_lines
    latest_lines.clear()
    middle_x = sum(sx) / len(sx)
    middle_y = sum(sy) / len(sy)
    color = colors[randint(0, len(colors) - 1)]
    for i in range(len(sx)):
        lines = ax.plot([middle_x, sx[i]], [middle_y, sy[i]])
        latest_lines.append(lines[0])
        pt = ax.scatter([middle_x], [middle_y], color=color)
        latest_lines.append(pt)


def random_lines():
    indexes = random.choices(range(len(x)), k=subset_size)
    sx = list(map(x.__getitem__, indexes))
    sy = list(map(y.__getitem__, indexes))
    plot_lines(sx, sy)


def specified_lines(subset_index):
    subsets = [
        {
            'x': [-6, 4, 5, -8],
            'y': [3, 2, -8, -9]
        },
        {
            'x': [0, 9, 9, 4],
            'y': [6, 6, 1, -3]
        },
        {
            'x': [4, 7, 10, 9],
            'y': [5, 8, 5, 2]
        },
        {
            'x': [8, 10, 9, 10],
            'y': [8, 8, 6, 6]
        },
    ]
    sx = subsets[subset_index]['x']
    sy = subsets[subset_index]['y']
    plot_lines(sx, sy)


def clear_plot():
    global latest_lines
    for l in latest_lines:
        l.remove()


def animate_random_lines(times=50):
    for i in range(times):
        random_lines()
        pause(1)
        clear_plot()


def animate_specified_lines(*lines):
    lines = lines or range(4)
    for l in lines:
        specified_lines(l)
        pause(2)
        clear_plot()


def cluster():
    rects = [
        patches.Rectangle((-6.5, 0), 5, 8, linewidth=2, edgecolor='r', facecolor='none'),
        patches.Rectangle((0, 0), 10, 8, linewidth=2, edgecolor='b', facecolor='none'),
        patches.Rectangle((-10, -10), 5, 9, linewidth=2, edgecolor='g', facecolor='none'),
        patches.Rectangle((3, -10), 7, 7, linewidth=2, edgecolor='c', facecolor='none'),
    ]
    [ax.add_patch(rect) for rect in rects]


def random_lines_of_cluster():
    clusters = [
        {
            'xs': [-5, -5, -6, -6, -4, -2, -2],
            'ys': [7, 5, 4, 2, 1, 4, 6]
        }, {
            'xs': [1, 3, 4, 8, 9, 8, 7],
            'ys': [7, 6, 2, 1, 6, 8, 8]
        }, {
            'xs': [-5, -8, -9, -10, -10, -8],
            'ys': [-3, -9, -9, -7, -3, -3]
        }, {
            'xs': [3, 3, 3, 4, 4, 4, 7, 8, 9, 9, 10],
            'ys': [-3, -6, -9, -3, -5, -7, -8, -9, -6, -8, -5]
        }
    ]
    xs = []
    ys = []
    for i in range(4):
        cls = clusters[i]
        pt = randint(0, len(cls['xs']) - 1)
        xs.append(cls['xs'][pt])
        ys.append(cls['ys'][pt])
    plot_lines(xs, ys)


def animate_random_lines_of_cluster(times=50):
    for i in range(times):
        random_lines_of_cluster()
        pause(1)
        clear_plot()

generate_points()
ax.scatter(x, y)
pause(1)
cluster()
pause(1)
animate_random_lines_of_cluster()
plt.show()
