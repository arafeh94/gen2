import json
import typing

import numpy as np
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB
from src.apis.utils import str_all_in

graphs = Graphs(FedDB('../fed_runs/seqfed2.sqlite'))

ax_configs = {
    'seqop_rn': {'color': '#ff7f0e', 'label': 'Seq_RN', 'linestyle': "--", 'linewidth': 2},
    'seqop_ga': {'color': '#1f77b4', 'label': 'Seq_GA', 'linestyle': "--", 'linewidth': 2},
    'seqop_all': {'color': '#7f7f7f', 'label': 'Seq_All', 'linestyle': "--", 'linewidth': 2},
    'ewc_rn': {'color': '#ff7f0e', 'label': 'EWC_RN', 'linestyle': "-", 'linewidth': 2},
    'ewc_ga': {'color': '#1f77b4', 'label': 'EWC_GA', 'linestyle': "-", 'linewidth': 2},
    'ewc_all': {'color': '#7f7f7f', 'label': 'EWC_All', 'linestyle': "-", 'linewidth': 2},
    'warmup': {'color': 'red', 'label': 'Shared', 'linestyle': "-", 'linewidth': 2},
    'basic': {'color': 'k', 'label': 'Basic', 'linestyle': "-", 'linewidth': 2},
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
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=8, fontsize='x-large' if tts == 0 else 'large')
    plt.rcParams.update({'font.size': 14})
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
    def seq_ga():
        a = graphs.db().get('t1713441912', 'acc, time')
        a_acc = [t[0] for t in a]
        a_time = [t[1] for t in a]
        b = graphs.db().get('t1713479131', 'acc, time')
        b_acc = [t[0] for t in b]
        b_time = [t[1] / 10 for t in b]

        acc = a_acc + b_acc
        time = a_time + b_time
        return {'acc': acc, 'time': time, 'vertical_time': sum(a_time)}


    def fl():
        a = graphs.db().get('t1713465157', 'acc, time')
        a_acc = [t[0] for t in a]
        a_time = [t[1] / 10 for t in a]
        return {'acc': a_acc, 'time': a_time}


    def warmup():
        a = graphs.db().get('t1713577992', 'acc')
        a_acc = [t for t in a]
        a_time = [1 / 80] * len(a_acc)

        b = graphs.db().get('t1713640099', 'acc, time')
        b_acc = [t[0] for t in b]
        b_time = [t[1] / 10 for t in b]

        acc = a_acc + b_acc
        time = a_time + b_time
        return {'acc': acc, 'time': time}


    def warmup_fl_only():
        ts = ['t1713640099', 't1713649165', 't1713659499', 't1713680010']
        all_res_acc = []
        all_res_time = []
        for tt in ts:
            res = graphs.db().get(tt, 'acc, time')
            b_acc = [t[0] for t in res]
            b_time = [t[1] / 10 for t in res]
            all_res_acc.extend(b_acc)
            all_res_time.extend(b_time)
        return {'acc': all_res_acc, 'time': all_res_time}


    ga = seq_ga()
    fl = fl()
    war = warmup_fl_only()

    sessions1 = {
        't1': {'x': range(len(ga['acc'][:7000])), 'y': ga['acc'][:7000], 'config': ax_configs['seqop_ga']},
        't2': {'x': range(len(fl['acc'][:7000])), 'y': fl['acc'][:7000], 'config': ax_configs['basic']},
        't3': {'x': range(len(war['acc'][:7000])), 'y': war['acc'][:7000], 'config': ax_configs['warmup']}
    }

    sessions2 = {
        't1': {'x': np.cumsum(ga['time'][:7000]) / 1000, 'y': ga['acc'][:7000], 'config': ax_configs['seqop_ga']},
        't2': {'x': np.cumsum(fl['time'][:7000]) / 1000, 'y': fl['acc'][:7000], 'config': ax_configs['basic']},
        't3': {'x': np.cumsum(war['time'][:1500]) / 1000, 'y': war['acc'][:1500], 'config': ax_configs['warmup']}
    }


    def vertical_line(plot):

        plt_config(plot)
        plt.axvline(x=ga['vertical_time']/1000, color='black', linestyle='--', linewidth=3)
        plt.xlim(right=1.4)

        inset_ax = plt.axes((0.59, 0.15, 0.3, 0.3))  # Adjust these coordinates as needed
        inset_ax.plot(np.cumsum(war['time'][1500:7000]) / 1000, war['acc'][1500:7000], color='r')
        inset_ax.set_title('Shared Cont.')


    graphs.plot2(sessions2, plt_func=vertical_line)

# general notes:
# 1- ewc calculate fisher after every epoch, which means increasing the number of epochs exponentially affect the time
# 2- ewc works well when learn rate is low, need to check why it is bad when working in sequence
# 3- ewc works well on mnist, need to test different dataset and distributions
# 4- the number of rounds have no major effects on the accuracy as much as lr and epochs
