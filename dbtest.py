import pymongo
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor


count = 0

def _get():
    global count
    if count == 0:
        count = 1
        print("Excep")
        raise Exception('haha')
    print("OK")
    time.sleep(3)

def _handle_exception(loop, context):
    loop.run_in_executor(None, _get)

with ThreadPoolExecutor() as executor:
    loop = asyncio.get_event_loop()
    loop.set_default_executor(executor)
    loop.set_exception_handler(_handle_exception)
    loop.run_in_executor(executor, _get)




# client = pymongo.MongoClient()
# db = client.get_database('test')
# tt = db.get_collection('test_collection')
# tt.delete_many({})
# tt.insert_many([
#     {'txhash': 'a', 'start': 1, 'end': 5},
#     {'txhash': 'a', 'start': 3, 'end': 17},
#     {'txhash': 'a', 'start': -2, 'end': 644},
#     {'txhash': 'b', 'start': 6., 'end': 523},
#     # {'txhash': 'c', 'start': 1, 'end': 5},
# ])
#
# # tt.update_one(
# #     {'_id': 'a'},
# #     {'$set': {'start': 99}},
# #     upsert=True
# # )
# # print(list(tt.find({})))
#
# it = tt.aggregate([
#     {'$match': {'end': {'$lt': 600}}},
#     {'$group': {'_id': None, 'ok': {'$max': 'txhash'}}}
# ])
# print(list(it))

