import pymongo

client = pymongo.MongoClient()
db = client.get_database('test')
tt = db.get_collection('test_collection')
tt.delete_many({})
tt.insert_many([
    {'txhash': 'a', 'start': 1, 'end': 5},
    {'txhash': 'a', 'start': 3, 'end': 17},
    {'txhash': 'a', 'start': -2, 'end': 644},
    {'txhash': 'b', 'start': 6., 'end': 523},
    # {'txhash': 'c', 'start': 1, 'end': 5},
])

# tt.update_one(
#     {'_id': 'a'},
#     {'$set': {'start': 99}},
#     upsert=True
# )
# print(list(tt.find({})))

it = tt.aggregate([
    {'$match': {'end': {'$lt': 600}}},
    {'$group': {'_id': None, 'ok': {'$max': 'txhash'}}}
])
print(list(it))

