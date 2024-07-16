import sqlite3


class FedDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.con = sqlite3.connect(db_path)

    def execute(self, query, params=None):
        cur = self.con.cursor()
        if params:
            return cur.execute(query, params)
        else:
            return cur.execute(query)

    def session_id(self, **tags):
        tables = self.get('session', '*')

    def get(self, table_name, field, where=''):
        where = where or ''
        query = f'select {field} from {table_name} {where}'
        return self.query(query)

    def query(self, q):
        records = self.execute(q)
        values = list(map(lambda x: x[0] if len(x) == 1 else x, records))
        return values

    def acc(self, table_name):
        query = f'select acc from {table_name}'
        records = self.execute(query)
        acc = list(map(lambda x: x[0], records))
        return acc

    def table_exists(self, table_name):
        cursor = self.con.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        result = cursor.fetchone()
        cursor.close()
        return result is not None

    def clean(self):
        sessions = self.tables()
        for session in sessions:
            if not self.table_exists(session):
                self.execute("DELETE FROM session WHERE session_id=?;", (session,))
                self.con.commit()

    def tables(self):
        query = 'select * from session'
        records = self.execute(query)
        records = list(map(lambda x: {x[0]: x[1]}, records))
        tables = {}
        for r in records:
            tables.update(r)
        return tables

    def merge(self, fed_db: 'FedDB'):
        self.execute(f'ATTACH "{fed_db.db_path}" AS db2')
        q1 = 'select session_id, config from session'
        q2 = 'CREATE TABLE if not exists [t] AS SELECT * FROM db2.[t];'
        q3 = 'insert or ignore into session values (?,?)'
        tables = fed_db.execute(q1)
        for table in tables:
            session_id, config = table
            create_table = q2.replace('[t]', session_id)
            print(f'executing {create_table}')
            self.execute(create_table)
            self.execute(q3, [session_id, config])
        self.con.commit()
