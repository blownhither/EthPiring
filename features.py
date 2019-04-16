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


def get_transactions_by_range(start=None, end=None):
    """
    Fetch all transactions from w3/db whose tx_hash is found in db with starting time between param `start` and `end`
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

    if len(hashes) == 0:
        return []

    transactions = multithread_get_transactions(hashes)
    transactions = [_extract_keys(x) for x in transactions if x is not None]
    for t in transactions:
        h = t['hash']
        t['first'] = first.get(h, None)
        t['last'] = last.get(h, None)

    print('Found {} valid transactions'.format(len(transactions)), flush=True)
    return transactions


def main():
    init()

    start = 1554961692.21946
    # end = 1554961732.21946
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

