import typing
from collections import defaultdict
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt, pyplot

from src.apis.fed_sqlite import FedDB


class Graphs:
    def __init__(self, db: FedDB):
        self._db = db

    def db(self):
        return self._db

    def __repr__(self):
        tables = self._db.tables()
        rpr = f'columns: session;config\n'
        for session, config in tables.items():
            rpr += f'{session} \t | \t {config}\n'
        rpr = rpr.rstrip('\n')
        return rpr

    def _as_dict(self, keys, default_value=dict):
        if isinstance(keys, dict):
            return keys
        res = {}
        for key in keys:
            res[key] = default_value(key)
        return res

    def plot(self, configs: list, title='', animated=False, save_path='', xlabel='', ylabel='',
             plt_func: Callable[[pyplot], typing.NoReturn] = None, show=True):
        """
        Args:
            plt_func:
            xlabel:
            ylabel:
            configs: a array of dictionaries containing session_id: the session id in the database,
                field: the field name in the table,
                config: the plot configurations,
                transform: a callable to transform the values to another, take values as input
                where: add where to the query "where a=1 and b=2"
            title: the title of the plot
            animated: animate the image (require the normal.py plot to be shown not the one in intellij
            save_path: save location if needed
            example:
            graphs.plot([
                {
                    'session_id': 'dbs_table_name',
                    'field': 'dbs_table_field_name',
                    'config': {'color': 'b'},
                    'transform': some_transformation_function
                },
            ])
        """
        plt.clf()
        sessions = [(
            item['session_id'],
            item['field'] if 'field' in item else None,
            item['config'] if 'config' in item else {},
            item['transform'] if 'transform' in item else None,
            item['where'] if 'where' in item else None,
            item['query'] if 'query' in item else None,
        ) for item in configs]
        session_values = {}
        for session_id, field, config, transform, where, query in sessions:
            values = self._db.query(query) if query else self._db.get(session_id, field, where)
            if transform:
                transformers = transform if isinstance(transform, list) else [transform]
                for trans in transformers:
                    values = trans(values)
            print(values)
            session_values[f'{session_id}_{field}_{str(transform)}'] = values
        if animated:
            pause = animated if isinstance(animated, (int, float)) else 0.05
            session_end = [False] * len(sessions)
            round_id = 0
            session_plot_values = defaultdict(list)
            while False in session_end:
                for session_id, field, config, transform in sessions:
                    try:
                        session_plot_values[f'{session_id}_{field}_{str(transform)}'].append(
                            session_values[f'{session_id}_{field}_{str(transform)}'][round_id])
                        plt.plot(session_plot_values[f'{session_id}_{field}_{str(transform)}'], **config)
                    except IndexError as e:
                        session_end[session_end.index(False)] = True
                plt.pause(pause)
                round_id += 1
        else:
            for session_id, field, config, transform, where, query in sessions:
                plot_vals = np.array(session_values[f'{session_id}_{field}_{str(transform)}'])
                plt.plot(plot_vals, **config)
        if callable(plt_func):
            plt_func(plt)
        plt.xlabel(xlabel, fontsize='large', labelpad=5)
        plt.ylabel(ylabel, fontsize='large', labelpad=5)
        fig = plt.gcf()
        fig.set_size_inches(16, 8)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=100)
        if show:
            plt.show()
        return plt

    def plot2(self, sessions, title='', save_path='', xlabel='', ylabel='',
              plt_func: Callable[[pyplot], typing.NoReturn] = None, show=True):
        plt.clf()
        for session_id, vals in sessions.items():
            plt.plot(vals['x'], vals['y'], **vals['config'] if 'config' in vals else {})

        if callable(plt_func):
            plt_func(plt)
        plt.xlabel(xlabel, fontsize='large', labelpad=5)
        plt.ylabel(ylabel, fontsize='large', labelpad=5)
        fig = plt.gcf()
        fig.set_size_inches(16, 8)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=100)
        if show:
            plt.show()
        return plt

