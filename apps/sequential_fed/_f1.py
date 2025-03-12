import json

from src.apis.fed_sqlite import FedDB


def decode(row):
    res = {'acc': row[1], 'loss': row[2]}
    for in_item in row:
        if 'accuracy' in str(in_item) and 'precision' in str(in_item):
            ob = json.loads(in_item.replace("'", '"'))
            res['precision'] = ob['precision']
            res['recall'] = ob['recall']
            res['f1'] = ob['f1']
    return res


db = FedDB('./seqfed3.sqlite')
res = {}
for key, item in db.tables().items():
    row = db.query(f'SELECT * FROM {key} order by round_id desc limit 1')[0]
    result = decode(row)
    try:
        time_taken = db.query(f'select sum(time) from {key}')[0]
    except Exception as e:
        time_taken = db.query(f'select time_taken from {key} limit 1')[0]
    result['time_taken'] = time_taken
    result['config'] = json.loads(item.replace("'", '"'))
    res[key] = result

print(res)


