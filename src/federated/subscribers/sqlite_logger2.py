import calendar
import logging
import sqlite3
import time

from src.apis import utils
from src.federated.events import FederatedSubscriber
from src.federated.federated import FederatedLearning


# noinspection SqlNoDataSourceInspection,SqlDialectInspection
class SQLiteLogger(FederatedSubscriber):
    def __init__(self, id, db_path, config=''):
        super().__init__()
        self.id = id
        utils.validate_path(db_path)
        self.con = sqlite3.connect(db_path, timeout=1000)
        self.initialized = False
        self._logger = logging.getLogger('sqlite')
        self.tag = str(config)
        self._check_table_name()
        self._init()

    def _init(self):
        query = 'create table if not exists session (session_id text primary key, config text)'
        self._execute(query)
        query = f"insert or replace into session values (?,?)"
        self._execute(query, [self.id, self.tag])

    def _create_table(self, **kwargs):
        if not self.initialized:
            params = self._extract_params(**kwargs)
            sub_query = ''
            for param in params:
                sub_query += f'{param[0]} {param[1]},'
            sub_query = sub_query.rstrip(',')
            query = f'''
            create table if not exists {self.id} (
                {sub_query}
            )
            '''
            self._execute(query)
            self.initialized = True

    def _insert(self, params):
        sub_query = ' '.join(['?,' for _ in range(len(params))]).rstrip(',')
        query = f'insert OR replace into {self.id} values ({sub_query})'
        values = list(map(lambda v: str(v) if isinstance(v, (list, dict)) else v, params.values()))
        self._execute(query, values)

    def _execute(self, query, params=None):
        cursor = self.con.cursor()
        self._logger.debug(f'executing {query}')
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        self.con.commit()

    def close(self):
        self.con.close()

    def _extract_params(self, **kwargs):
        def param_map(val):
            if isinstance(val, int):
                return 'INTEGER'
            elif isinstance(val, str):
                return 'TEXT'
            elif isinstance(val, float):
                return 'FLOAT'
            else:
                return 'text'

        params = [('round_id', 'INTEGER PRIMARY KEY')]
        for key, val in kwargs.items():
            params.append((key, param_map(val)))
        return params

    def log(self, round_id, **kwargs):
        self._create_table(**kwargs)
        record = {'round_id': round_id, **kwargs}
        self._insert(record)

    def log_all(self, round_id, args: dict):
        self._create_table(**args)
        record = {'round_id': round_id, **args}
        self._insert(record)

    def on_round_end(self, params):
        context: FederatedLearning.Context = params['context']
        last_record: dict = context.history[context.round_id]
        self.log(context.round_id, **last_record)

    def _check_table_name(self):
        if self.id is None:
            self.id = 'None'
        if self.id[0].isdigit():
            self.id = f't{self.id}'

    @staticmethod
    def new_instance(path, configs):
        return SQLiteLogger(str(calendar.timegm(time.gmtime())), path, configs)
