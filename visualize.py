import time
import pymongo
import numpy as np
import asyncio

from build import get_transaction, multithread_get_transactions
from matplotlib import pyplot as plt



a = np.load(open('records.npy', 'rb'))
plt.hist(a, bins=300)
plt.show()
exit()


HOUR = 3600

client = pymongo.MongoClient()
db = client.get_database('eth')
tt = db.get_collection('transaction_timestamp')
it = tt.find({'start': {'$gt': time.time() - 4.8 * HOUR, '$lt': time.time() - 4 * HOUR}}).distinct('txhash')



records = []
trans = multithread_get_transactions(list(it))
for t in trans:
    if t is None:
        print("None")
        continue
    price = t['gasPrice']
    print(price)
    records.append(price)

# for tx_hash in it:
#     print(tx_hash)
#     transaction = get_transaction(tx_hash)
#     if transaction is None:
#         continue
#     price = transaction['gasPrice']
#     print(price)
#     records.append(price)

np.save('records.npy', records)
plt.hist(records)
plt.yscale('log')
plt.savefig('hist.png')
