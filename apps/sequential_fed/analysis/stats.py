import json
import typing

import numpy as np
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB
from src.apis.utils import str_all_in

graphs = Graphs(FedDB('../seqfed.sqlite'))

ax_configs = {
    'seqop_rn': {'color': '#ff7f0e', 'label': 'WFSL_RN', 'linestyle': "--", 'linewidth': 3},
    'seqop_ga': {'color': '#1f77b4', 'label': 'WFSL_GA', 'linestyle': "--", 'linewidth': 3},
    'seqop_all': {'color': '#7f7f7f', 'label': 'WFSL_All', 'linestyle': "--", 'linewidth': 3},
    'ewc_rn': {'color': '#ff7f0e', 'label': 'EWC_RN', 'linestyle': "-", 'linewidth': 3},
    'ewc_ga': {'color': '#1f77b4', 'label': 'EWC_GA', 'linestyle': "-", 'linewidth': 3},
    'ewc_all': {'color': '#7f7f7f', 'label': 'EWC_All', 'linestyle': "-", 'linewidth': 3},
    'warmup': {'color': 'red', 'label': 'Shared', 'linestyle': "-", 'linewidth': 3},
    'default': {'color': 'k', 'label': 'Any', 'linestyle': "-", 'linewidth': 3},
}


class C:
    @staticmethod
    def mnist(id):
        return 'mnist' in id

    @staticmethod
    def kdd(id):
        return 'kdd' in id

    @staticmethod
    def shard(id):
        return 'shard' in id


def log_transform(dt):
    dt = np.array(dt)
    non_zero_mask = (dt != 0)
    result = np.where(non_zero_mask, np.log10(dt), 0)
    return result


tts = 0


def plt_config(plt):
    global tts
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=8, fontsize='18' if tts == 0 else '18')
    plt.rcParams.update({'font.size': 16})
    plt.gca().tick_params(axis='both', which='major', labelsize='large')
    plt.xlim(0)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('gray')
    tts += 1


def acc(sessions, trans=None, save_path=None):
    configurations = []
    for exp_code, session in sessions.items():
        configurations.append({
            'session_id': session,
            'transform': trans,
            'field': 'acc',
            'config': ax_configs[exp_code]
        })
    graphs.plot(configurations, 'Plot', xlabel='Rounds', ylabel='Accuracy', plt_func=plt_config, save_path=save_path)


def time2acc(sessions, trans=None, save_path=None):
    sessions_val = {}
    for ex_code, table in sessions.items():
        data = graphs.db().query(f"select acc, time from {table}")
        acc_dt = [0] + [d[0] for d in data]
        acc_dt = trans(acc_dt) if trans else acc_dt
        time_dt = [0] + [d[1] / 1000 for d in data]
        time_cumulative = np.cumsum(time_dt)
        sessions_val[table] = {'x': time_cumulative, 'y': acc_dt, 'config': ax_configs[ex_code]}
    graphs.plot2(sessions_val, 'time2acc', xlabel='Cumulative Time', ylabel='Accuracy',
                 plt_func=plt_config, save_path=save_path)


def generate_exps(session_ids, filter=None):
    table_names = ', '.join(map(lambda x: str(f"'{x}'"), session_ids))
    query = f"SELECT * FROM session WHERE session_id IN ({table_names})"
    sess = graphs.db().query(query)
    res = {}
    for item in sess:
        sess_item = edict(json.loads(item[1].replace("'", '"')))
        print(item[0] + ": ", item[1])
        name = sess_item['method']
        if (filter and filter(name, sess_item)) or not filter:
            res[name] = item[0]
    return res


def get_value(dictionary, key_path, default=None):
    keys = key_path.split('.')
    value = dictionary
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            value = default
            break
    return value


def check(arg_value, value):
    if isinstance(arg_value, list):
        return all([check(a, value) for a in arg_value])
    if callable(arg_value):
        if not arg_value(value):
            return False
    elif value != arg_value:
        return False
    return True


def collect(conditions: typing.Union[list, dict]) -> list:
    if isinstance(conditions, list):
        sessions = []
        for item in conditions:
            sessions.extend(collect(item))
        return sessions

    session_ids = []
    query = f"SELECT * FROM session"
    sess = graphs.db().query(query)
    for sess_id, conf in sess:
        conf = edict(json.loads(conf.replace("'", '"')))
        accepted = True
        for arg_name, arg_value in conditions.items():
            value = get_value(conf, arg_name)
            if not check(arg_value, value):
                accepted = False
        if accepted:
            session_ids.append(sess_id)
    return session_ids


def delete(session_ids):
    for sess_id in session_ids:
        graphs.db().query(f'drop table if exists {sess_id}')
        graphs.db().query(f'delete from session where session_id = "{sess_id}"')
        graphs.db().con.commit()


def standard(ss, filter=None, save=None):
    acc(generate_exps(ss, filter), utils.smooth, save_path=f'../plots/{save}.acc.png' if save else None)
    time2acc(generate_exps(ss, filter), utils.smooth, save_path=f'../plots/{save}.time.png' if save else None)


def wmp(sessions, trans=None, save_path=None):
    configurations = []
    colors = ['#ff7f0e', '#7f7f7f', '#1f77b4', 'k']
    lines = ['-', '--', ':', '-.']
    for index, session in enumerate(sessions):
        print(session)
        sn = {'t1710445667': 'Shard_KDD', 't1710445719': 'Dir_KDD',
              't1710445766': 'Shard_Mnist', 't1710445793': 'Dir_Mnist'}
        configurations.append({
            'session_id': session,
            'transform': trans,
            'field': 'test_acc',
            'config': {'color': colors[index % len(colors)], 'label': sn[session],
                       'linestyle': lines[index % len(lines)], 'linewidth': 2.5}
        })
    graphs.plot(configurations, 'Plot', xlabel='Rounds', ylabel='Accuracy', plt_func=plt_config, save_path=save_path)


if __name__ == '__main__':
    # graphs.db().clean()
    # good experiments to represent:
    # ==============================
    # ------ for mnist shard
    # runs = collect({'id': C.mnist, 'wmp.rounds': 50, 'wmp.epochs': 20, 'wmp.lr': 0.01, 'method': lambda x: x != 'ewc_ga'})
    # ------ mnist dirichlet
    # runs = collect({'tag': '15', 'method': lambda x: x != 'ewc_ga'})

    # ------ for kdd shard
    # runs = collect({'tag': '32', 'method': lambda x: x != 'ewc_ga'})
    # runs = [r for r in runs if r != 't1710215848']

    # ------ kdd dirichlet
    runs = collect({'tag': 'pb3', 'method': lambda x: x != 'ewc_ga'})

    # wmp(collect({'tag': 'wmp01'}), utils.smooth)
    # standard(collect({'tag': '32'}), lambda name, item: 'warmup' not in name and 'ewc_ga' not in name, save='kdd_shard')
    # standard(collect({'tag': 'pb3'}), lambda name, item: 'warmup' not in name and 'ewc_ga' not in name, save='kdd_dir')
    # standard(collect({'id': C.mnist, 'wmp.rounds': 50, 'wmp.epochs': 20, 'wmp.lr': 0.01}),
    #          lambda name, item: 'warmup' not in name and 'ewc_ga' not in name, save='mnist_shard')
    # standard(collect({'tag': '15'}), lambda name, item: 'warmup' not in name and 'ewc_ga' not in name, save='mnist_dir')
    standard(runs)


# general notes:
# 1- ewc calculate fisher after every epoch, which means increasing the number of epochs exponentially affect the time
# 2- ewc works well when learn rate is low, need to check why it is bad when working in sequence
# 3- ewc works well on mnist, need to test different dataset and distributions
# 4- the number of rounds have no major effects on the accuracy as much as lr and epochs
