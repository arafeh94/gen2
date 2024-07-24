from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graphs = Graphs(FedDB('splitlearn_v2.sqlite'))

s = graphs.db().query('select * from session where config like "%split.py%"')
s = [t[0] for t in s]
for ss in s:
    try:
        print(ss, graphs.db().query(f"select count(*) from {ss}"))
    except Exception as e:
        pass