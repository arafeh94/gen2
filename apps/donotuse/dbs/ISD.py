from src.apis.fed_sqlite import FedDB

db = FedDB('res.db')

print(db.tables())
print(db.get(table_name='iid', field='local_acc')[0])
