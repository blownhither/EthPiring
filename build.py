import os
import time
import datetime
import asyncio
from timeit import default_timer
from concurrent.futures import ThreadPoolExecutor


os.environ['WEB3_INFURA_API_KEY'] = '192d2f19099e4ec188ba2492d8928bdc'
from web3.auto.infura import w3
import database as db


def init():
    if not w3.isConnected():
        raise Exception('Not connected')


def get_pending_transactions():
    return w3.eth.getBlock('pending')['transactions']


def get_transaction(tx_hash):
    """
    get transaction from w3, or db
    """
    ret = db.get_transaction(tx_hash)
    if ret is not None:
        return ret
    ret = w3.eth.getTransaction(tx_hash)
    if ret is not None:
        db.insert_transaction(dict(ret))
    print('get_transaction:', tx_hash[:10])
    return ret


def multithread_get_transactions(tx_hashes):
    async def _work():
        with ThreadPoolExecutor(max_workers=128) as executor:
            loop = asyncio.get_event_loop()
            tasks = [loop.run_in_executor(executor, get_transaction, h) for h in tx_hashes]
            return [response for response in await asyncio.gather(*tasks)]

    responses = asyncio.run(_work())

    # # Python 3.6 implementation of asyncio.run()
    # # ('asyncio' has no attribute 'run' in Python 3.6)
    # loop = asyncio.get_event_loop()
    # responses = loop.run_until_complete(_work())
    # loop.close()

    return [dict(x) for x in responses if x is not None]


class RollingPool:
    def __init__(self):
        self.transaction_pool = set()
        self.timestamp = {}

    def update(self, current, timestamp=None):
        """
        update the pool with some new transactions. Timestamps are recorded for the new comers, gone transactions
        are returned
        :param current: List(HexBytes)
        :param timestamp: float|None
        :return: List(Tuple(hex_str, float start, float end)), transactions no longer pending
        """
        if timestamp is None:
            timestamp = time.time()
        current = set(current)
        new_tx = current - self.transaction_pool
        for t in new_tx:
            self.timestamp[t] = timestamp
        submitted_transaction = self.transaction_pool - current
        self.transaction_pool = current
        ret = [(s.hex(), self.timestamp[s], timestamp) for s in submitted_transaction]
        for s in submitted_transaction:
            self.timestamp.pop(s)
        return ret

    def record(self, entries):
        """
        record 3-tuple into db, ['txhash', 'start', 'end']
        :param entries:
        :return:
        """
        count = None
        for ent in entries:
            count = db.insert_one({
                'txhash': ent[0],
                'start': ent[1],
                'end': ent[2],
            })
        return count

    def run_forever(self, duration=1):
        while True:
            try:
                transactions = get_pending_transactions()
            except Exception as e:
                print('Exception: ', e)
                time.sleep(duration)
                continue
            submitted = self.update(transactions)
            count = self.record(submitted)
            print(datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                  ' pending: ', len(transactions),
                  ' submitted: ', len(submitted),
                  ' db_count: ', count, flush=True)
            time.sleep(duration)


def main():
    init()
    p = RollingPool()
    p.run_forever(1)


if __name__ == '__main__':
    main()


