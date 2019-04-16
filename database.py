import sys
from pymongo import MongoClient
import threading


_thread_local = threading.local()


def _get_thread_collection_session():
    coll = getattr(_thread_local, 'db_tx_collection_session', None)
    if coll is None:
        coll = MongoClient().get_database('eth').get_collection('transactions')
        _thread_local.db_tx_collection_session = coll
    return coll


def insert_transaction(transaction):
    transaction['hash'] = transaction['hash'].hex()
    transaction['value'] = float(transaction['value'])
    transaction['gasPrice'] = float(transaction['gasPrice'])
    _get_thread_collection_session().insert_one(transaction)


def get_transaction(h):
    ret = _get_thread_collection_session().find_one({'hash': h})
    return ret


def get_timestamp_collection():
    client = MongoClient()
    db = client.get_database('eth')
    tt = db.get_collection('transaction_timestamp')
    return tt


def get_database():
    client = MongoClient()
    db = client.get_database('eth')
    return db


def insert_one(data):
    client = MongoClient()
    db = client.get_database('eth')
    tt = db.get_collection('transaction_timestamp')
    tt.insert_one(data)
    return tt.estimated_document_count()


def integrity_check():
    client = MongoClient()
    db = client.get_database('eth')
    tt = db.get_collection('transaction_timestamp')
    data = list(tt.find())
    # print(data[0])
    hashes = [row['txhash'] for row in data]
    from collections import Counter
    dup = Counter(hashes).most_common(10)
    # print(dup[0])
    s = [row for row in data if row['txhash'] == dup[0][0]]
    # print(s)


def clean():
    client = MongoClient()
    db = client.get_database('eth')
    tt = db.get_collection('transaction_timestamp')
    print('count: ', tt.estimated_document_count())
    ret = tt.delete_many({})
    print(ret)
    print('count: ', tt.estimated_document_count())


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'clean':
        ret = input('Are you sure you want to delete everything?')
        if ret == 'y':
            clean()
        else:
            print("Nope")



