import json
import typing

import numpy as np
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graphs = Graphs(FedDB('splitlearn_v2_1.sqlite'))

ax_configs = {
    './sfed.py': {'color': 'm', 'label': 'Fed', 'linestyle': "-", 'linewidth': 3.2},
    './split.py': {'color': 'r', 'label': 'SL', 'linestyle': "-.", 'linewidth': 3.2},
    './splitfed.py': {'color': 'b', 'label': 'SFL', 'linestyle': ":", 'linewidth': 3.2},
    './splitfed1layer.py': {'color': 'g', 'label': '1L', 'linestyle': ":", 'linewidth': 3.2},
    './splitfed2layers_selection.py': {'color': 'y', 'label': '2Lo', 'linestyle': "-", 'linewidth': 3.2},
    './splitfed2layers_selection_v1.py': {'color': 'm', 'label': '2Lo1', 'linestyle': "-", 'linewidth': 3.2},
    './splitfed2layers_standard.py': {'color': 'c', 'label': '2Ls', 'linestyle': "--", 'linewidth': 3.2},
    './splitfed2layers_standard_v1.py': {'color': 'k', 'label': '2Ls1', 'linestyle': "--", 'linewidth': 3.2},
}


def log_transform(dt):
    dt = np.array(dt)
    non_zero_mask = (dt != 0)
    result = np.where(non_zero_mask, np.log10(dt), 0)
    return result


def plt_config(plt):
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=8, fontsize='large')
    plt.rcParams.update({'font.size': 12})
    plt.gca().tick_params(axis='both', which='major', labelsize='large')
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('gray')


def acc(sessions, trans=None):
    configurations = []
    for exp_code, session in sessions.items():
        configurations.append({
            'session_id': session,
            'transform': trans,
            'field': 'round_time',
            'query': f'select max(acc) as acc from {session} group by round_num',
            'config': ax_configs[exp_code]
        })
    graphs.plot(configurations, 'Plot', xlabel='Rounds', ylabel='Accuracy', plt_func=plt_config)


def time(sessions, trans=None):
    configurations = []
    for exp_code, session in sessions.items():
        configurations.append({
            'session_id': session,
            'transform': trans,
            'field': 'round_time',
            'query': f'select max(round_time) as round_time from {session} group by round_num',
            'config': ax_configs[exp_code]
        })
    graphs.plot(configurations, 'Plot', xlabel='Rounds', ylabel='Exec Time', plt_func=plt_config)


def time2acc(sessions, trans=None):
    sessions_val = {}
    for ex_code, table in sessions.items():
        query_str = f"select max(acc) as acc, max(round_time) as round_time from {table} group by round_num"
        data = graphs.db().query(query_str)
        acc_dt = [0] + [d[0] for d in data]
        acc_dt = trans(acc_dt) if trans else acc_dt
        time_dt = [0] + [d[1] / 1000 for d in data]
        time_cumulative = np.cumsum(time_dt)
        sessions_val[table] = {'x': time_cumulative, 'y': acc_dt, 'config': ax_configs[ex_code]}
    graphs.plot2(sessions_val, 'time2acc', xlabel='Cumulative Time', ylabel='Accuracy',
                 plt_func=plt_config)


def generate_exps(session_ids, filter=None):
    table_names = ', '.join(map(lambda x: str(f"'{x}'"), session_ids))
    query = f"SELECT * FROM session WHERE session_id IN ({table_names})"
    sess = graphs.db().query(query)
    res = {}
    for item in sess:
        sess_item = edict(json.loads(item[1].replace("'", '"')))
        print(item[0] + ": ", item[1])
        name = sess_item['name']
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


def v0v1(ss):
    selection = ['2layers_selection', '2layers_selection_v1', '2layers_standard', '2layers_standard_v1']
    acc(generate_exps(ss, selection), utils.smooth)
    time2acc(generate_exps(ss, selection), utils.smooth)
    time(generate_exps(ss, selection), utils.smooth)


def deny(items):
    def s(_, content):
        for item in items:
            if item in content['name']:
                return False
        return True

    return s


def standard(ss):
    acc(generate_exps(ss), utils.smooth)
    time2acc(generate_exps(ss), utils.smooth)
    time(generate_exps(ss), utils.smooth)


if __name__ == '__main__':
    code_exp = ['split', 'splitfed', '1layer', '2layers_selection', '2layers_selection_v1',
                '2layers_standard', '2layers_standard_v1']
    # ss = collect({'tag': 'exp1', 'name': lambda x: x != './split.py'})
    ss = collect({'tag': 'exp1'})

    standard(ss)
