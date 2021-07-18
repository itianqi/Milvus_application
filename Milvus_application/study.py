import numpy as np
import random
from milvus import Milvus
from milvus import Status

_HOST = '192.168.xx.xx'
_PORT = 19530

# Connect to Milvus Server
milvus = Milvus(_HOST, _PORT)

# Close client instance
# milvus.close()

# Returns the status of the Milvus server.
server_status = milvus.server_status(timeout=4)
print(server_status)


# Vector parameters
_DIM = 8  # dimension of vector

_INDEX_FILE_SIZE = 32  # max file size of stored index

# the demo name.
collection_name = 'example_collection_'
partition_tag = 'demo_tag_'
segment_name= ''

# 10 vectors with 8 dimension, per element is float32 type, vectors should be a 2-D array
vectors = [[random.random() for _ in range(_DIM)] for _ in range(10)]
ids = [i for i in range(10)]

print(vectors)

# Returns the version of the client.
client_version= milvus.client_version()
print(client_version)

# Returns the version of the Milvus server.
server_version = milvus.server_version(timeout=10)
print(server_version)

print("has collection:",milvus.has_collection(collection_name=collection_name, timeout=10))


from milvus import DataType
# Information needed to create a collection.Defult index_file_size=1024 and metric_type=MetricType.L2
collection_param = {
    "fields": [
        #  Milvus doesn't support string type now, but we are considering supporting it soon.
        #  {"name": "title", "type": DataType.STRING},
        {"name": "duration", "type": DataType.INT32, "params": {"unit": "minute"}},
        {"name": "release_year", "type": DataType.INT32},
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 8}},
    ],
    "segment_row_limit": 4096,
    "auto_id": False
}

# ------
# Basic create collection:
#     After create collection `demo_films`, we create a partition tagged "American", it means the films we
#     will be inserted are from American.
# ------
# milvus.create_collection(collection_name, collection_param)
# milvus.create_partition(collection_name, "American")


# ------
# Basic create collection:
#     You can check the collection info and partitions we've created by `get_collection_info` and
#     `list_partitions`
# ------
print("--------get collection info--------")
collection = milvus.get_collection_info(collection_name)
print(collection)
partitions = milvus.list_partitions(collection_name)
print("\n----------list partitions----------")
print(partitions)

# ------
# Basic insert entities:
#     We have three films of The_Lord_of_the_Rings series here with their id, duration release_year
#     and fake embeddings to be inserted. They are listed below to give you a overview of the structure.
# ------
The_Lord_of_the_Rings = [
    {
        "title": "The_Fellowship_of_the_Ring",
        "id": 1,
        "duration": 208,
        "release_year": 2001,
        "embedding": [random.random() for _ in range(8)]
    },
    {
        "title": "The_Two_Towers",
        "id": 2,
        "duration": 226,
        "release_year": 2002,
        "embedding": [random.random() for _ in range(8)]
    },
    {
        "title": "The_Return_of_the_King",
        "id": 3,
        "duration": 252,
        "release_year": 2003,
        "embedding": [random.random() for _ in range(8)]
    }
]

# ------
# Basic insert entities:
#     To insert these films into Milvus, we have to group values from the same field together like below.
#     Then these grouped data are used to create `hybrid_entities`.
# ------
ids = [k.get("id") for k in The_Lord_of_the_Rings]
durations = [k.get("duration") for k in The_Lord_of_the_Rings]
release_years = [k.get("release_year") for k in The_Lord_of_the_Rings]
embeddings = [k.get("embedding") for k in The_Lord_of_the_Rings]

hybrid_entities = [
    # Milvus doesn't support string type yet, so we cannot insert "title".
    {"name": "duration", "values": durations, "type": DataType.INT32},
    {"name": "release_year", "values": release_years, "type": DataType.INT32},
    {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR},
]

# ------
# Basic insert entities:
#     We insert the `hybrid_entities` into our collection, into partition `American`, with ids we provide.
#     If succeed, ids we provide will be returned.
# ------
for _ in range(2000):
    ids = milvus.insert(collection_name, hybrid_entities, ids, partition_tag="American")
    print("\n----------insert----------")
    print("Films are inserted and the ids are: {}".format(ids))


# ------
# Basic insert entities:
#     After insert entities into collection, we need to flush collection to make sure its on disk,
#     so that we are able to retrieve it.
# ------
before_flush_counts = milvus.count_entities(collection_name)
milvus.flush([collection_name])
after_flush_counts = milvus.count_entities(collection_name)
print("\n----------flush----------")
print("There are {} films in collection `{}` before flush".format(before_flush_counts, collection_name))
print("There are {} films in collection `{}` after flush".format(after_flush_counts, collection_name))

# ------
# Basic insert entities:
#     We can get the detail of collection statistics info by `get_collection_stats`
# ------
info = milvus.get_collection_stats(collection_name)
print("\n----------get collection stats----------")
print(info)

# ------
# Basic search entities:
#     Now that we have 3 films inserted into our collection, it's time to obtain them.
#     We can get films by ids, if milvus can't find entity for a given id, `None` will be returned.
#     In the case we provide below, we will only get 1 film with id=1 and the other is `None`
# ------
films = milvus.get_entity_by_id(collection_name, ids=[1, 200])
print("\n----------get entity by id = 1, id = 200----------")
for film in films:
    if film is not None:
        print(" > id: {},\n > duration: {}m,\n > release_years: {},\n > embedding: {}"
              .format(film.id, film.duration, film.release_year, film.embedding))

# ------
# Basic hybrid search entities:
#      Getting films by id is not enough, we are going to get films based on vector similarities.
#      Let's say we have a film with its `embedding` and we want to find `top3` films that are most similar
#      with it by L2 distance.
#      Other than vector similarities, we also want to obtain films that:
#        `released year` term in 2002 or 2003,
#        `duration` larger than 250 minutes.
#
#      Milvus provides Query DSL(Domain Specific Language) to support structured data filtering in queries.
#      For now milvus supports TermQuery and RangeQuery, they are structured as below.
#      For more information about the meaning and other options about "must" and "bool",
#      please refer to DSL chapter of our pymilvus documentation
#      (https://pymilvus.readthedocs.io/en/latest/).
# ------
query_embedding = [random.random() for _ in range(8)]
query_hybrid = {
    "bool": {
        "must": [
            {
                "term": {"release_year": [2002, 2003]}
            },
            {
                # "GT" for greater than
                "range": {"duration": {"GT": 250}}
            },
            {
                "vector": {
                    "embedding": {"topk": 3, "query": [query_embedding], "metric_type": "L2"}
                }
            }
        ]
    }
}

# ------
# Basic hybrid search entities:
#     And we want to get all the fields back in results, so fields = ["duration", "release_year", "embedding"].
#     If searching successfully, results will be returned.
#     `results` have `nq`(number of queries) separate results, since we only query for 1 film, The length of
#     `results` is 1.
#     We ask for top 3 in-return, but our condition is too strict while the database is too small, so we can
#     only get 1 film, which means length of `entities` in below is also 1.
#
#     Now we've gotten the results, and known it's a 1 x 1 structure, how can we get ids, distances and fields?
#     It's very simple, for every `topk_film`, it has three properties: `id, distance and entity`.
#     All fields are stored in `entity`, so you can finally obtain these data as below:
#     And the result should be film with id = 3.
# ------
results = milvus.search(collection_name, query_hybrid, fields=["duration", "release_year", "embedding"])
print("\n----------search----------")
for entities in results:
    for topk_film in entities:
        current_entity = topk_film.entity
        print("- id: {}".format(topk_film.id))
        print("- distance: {}".format(topk_film.distance))

        print("- release_year: {}".format(current_entity.release_year))
        print("- duration: {}".format(current_entity.duration))
        print("- embedding: {}".format(current_entity.embedding))

# ------
# Basic delete:
#     Now let's see how to delete things in Milvus.
#     You can simply delete entities by their ids.
# ------
# milvus.delete_entity_by_id(collection_name, ids=[1, 2])
# milvus.flush()  # flush is important
# result = milvus.get_entity_by_id(collection_name, ids=[1, 2])
#
# counts_delete = sum([1 for entity in result if entity is not None])
# counts_in_collection = milvus.count_entities(collection_name)
# print("\n----------delete id = 1, id = 2----------")
# print("Get {} entities by id 1, 2".format(counts_delete))
# print("There are {} entities after delete films with 1, 2".format(counts_in_collection))
#
# # ------
# # Basic delete:
# #     You can drop partitions we create, and drop the collection we create.
# # ------
# milvus.drop_partition(collection_name, partition_tag='American')
# if collection_name in milvus.list_collections():
#     milvus.drop_collection(collection_name)

# ------
# Summary:
#     Now we've went through all basic communications pymilvus can do with Milvus server, hope it's helpful!
# ------
#https://github.com/milvus-io/pymilvus/tree/0.3.0#insert-entities-in-a-collection


#建索引
ivf_param = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 4096}}
# the demo name.
collection_name = 'example_collection_'
partition_tag = 'demo_tag_'
segment_name= ''
_HOST = '192.168.xx.xx'
_PORT = 19530

# Connect to Milvus Server
client = Milvus(_HOST, _PORT)
client.create_index(collection_name, "embedding", ivf_param)


