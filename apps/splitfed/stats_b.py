import numpy as np
import matplotlib.pyplot as plt

from src.apis.fed_sqlite import FedDB

db = FedDB('./logs.db')
plt.rcParams.update({'font.size': 28})
plt.grid()

data = {}
tb = db.tables()
for t in tb:
    query = f"select sum(exec_time) from {t}"
    if t == 'main_splitfed_cluster_faster':
        query = f"select sum(exec_time) from {t} where cross=0"

    v = db.execute(query).fetchall()[0][0]
    if t == 'main_centra':
        continue
    if t == 'main_federated':
        continue
    if t == 'main_split':
        t = "Split"
    if t == 'main_splitfed':
        t = 'SplitFed'
    if t == 'main_splitfed_cluster':
        t = 'Cluster'
    if t == 'main_splitfed_cluster_faster':
        t = 'Ours'
    data[t] = v

courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(courses, values, width=0.4)

plt.xlabel("Approach")
plt.ylabel("Total Exec Time (s)")
plt.title("Total Exec Time Per Approach")
plt.show()
