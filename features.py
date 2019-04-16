import time
import datetime

import pandas as pds
import database
from build import w3, init, multithread_get_transactions
from collections import defaultdict
import util


def _yesterday_timestamp():
    """
    return start and end timestamp of yesterday
    """
    t = datetime.date.today() - datetime.timedelta(days=1)
    start = datetime.datetime(t.year, t.month, t.day).timestamp()
    t = datetime.date.today()
    end = datetime.datetime(t.year, t.month, t.day).timestamp()
    return start, end


def _end_of_day(ts):
    """
    return the timestamp by the end of that day (i.e. start of the next day)
    """
    next_day = datetime.date.fromtimestamp(ts) + datetime.timedelta(days=1)
    return datetime.datetime(next_day.year, next_day.month, next_day.day).timestamp()

#
# def multiprocess_w3_get_transactions(hashes):
#     keys = ['blockNumber', 'gas', 'gasPrice', 'transactionIndex', 'nonce', 'hash']
#
#     def _worker(hashes):
#         tt = database.get_collection()   # must not share client instance
#         for h in hashes:
#             tran = w3.eth.getTransaction(h)
#             if tran is None:
#                 print('get a None with ', h)
#                 continue
#             data = {k: tran[k] for k in keys}
#             tt.insert_one(data)
#
#     return util.balanced_parallel(_worker, hashes)


# def get_waiting_time(h, tt=None):
#     if tt is None:
#         tt = database.get_timestamp_collection()
#     it = tt.aggregate([
#         {'$match': {'txhash': h}},
#         {'$group': {'_id': None, 'tx_first': {'$min': '$start'}, 'tx_last': {'$max': '$end'}}}
#     ])
#     result = list(it)
#     if len(result) == 0:
#         return -1, -1, -1
#     result = list(it)[0]
#     first, last = result['tx_first'], result['tx_last']
#     return first, last, last - first


# def get_many_waiting_time(hashes):
#     tt = database.get_timestamp_collection()
#     return [get_waiting_time(h, tt) for h in hashes]


def get_transactions_by_range(start=None, end=None):
    """
    Fetch all transactions whose tx_hash is found in db with starting time between param `start` and `end`
    return w3 info, first and last
    :param start: timestamp, default to start of yesterday
    :param end: timestamp, default to start of today
    :return:
    """
    keys = ['blockNumber', 'gas', 'gasPrice', 'value', 'hash']

    def _extract_keys(d):
        ret = {k: d.get(k, '') for k in keys}
        return ret

    if start is None and end is None:
        start, end = _yesterday_timestamp()
    print('Get transactions between ',
          datetime.datetime.fromtimestamp(start),
          datetime.datetime.fromtimestamp(end))

    tt = database.get_timestamp_collection()

    future = time.time() * 10
    first = defaultdict(lambda: future)
    last = defaultdict(float)

    for tran in tt.find({'start': {'$gte': start, '$lt': end}}):
        h = tran['txhash']
        # print(type(h))
        first[h] = min(first[h], tran['start'])
        last[h] = max(last[h], tran['end'])

    hashes = first.keys()
    print('Found {} transactinos'.format(len(hashes)), flush=True)
    transactions = multithread_get_transactions(hashes)
    transactions = [_extract_keys(x) for x in transactions if x is not None]
    for t in transactions:
        h = t['hash'].hex()
        t['hash'] = h
        t['first'] = first.get(h, None)
        t['last'] = last.get(h, None)

    print('Found {} valid transactions'.format(len(transactions)), flush=True)
    return transactions


# def insert_waiting_time(transactions):
#     hashes = [x['hash'] for x in transactions]
#     waiting = get_many_waiting_time(hashes)
#
#     db = database.get_database()
#     tx_collection = db.get_collection('transactions')
#     ret = tx_collection.insert_many(transactions)
#     print(ret)


def main():
    init()

    start = 1554961692.21946
    end = _end_of_day(start)

    transactions = get_transactions_by_range(start, end)
    df = pds.DataFrame(transactions)

    columns = list(df.columns)
    columns.remove('hash')
    columns = columns + ['hash']        # put has last

    name = datetime.datetime.fromtimestamp(start).strftime('%Y%m%d') + '.csv'
    print(name)
    df.to_csv(name, index=False, columns=columns)


if __name__ == '__main__':
    main()

