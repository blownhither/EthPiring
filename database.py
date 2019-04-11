import sys
from pymongo import MongoClient


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



