import random

import torch
import torch.nn as nn
import argparse
from utils import load_pretrained_embeddings, build_dataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from milvus import Milvus
from milvus import Status
from milvus import DataType


class BiLSTM(nn.Module):
    def __init__(self, embedding_pretrained, input_size, hidden_size, num_layers, ):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        
    def forward(self, x):
        batch_size = x.size(0)
        emb = self.embedding(x)
        out, (hn, cn) = self.lstm(emb)
        return hn.view(batch_size, -1)


def get_tensor(model, data_iter):
    res_tensor = []  # 存储文本对应的向量
    for i, (x, labels) in tqdm(enumerate(data_iter)):
        out = model(x)
        res_tensor.extend(out.data.numpy().tolist())
    return res_tensor


def get_text_tensor_pair(tensor_list, path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    assert len(lines) == len(tensor_list)
    pairs = []
    for line, tensor in zip(lines, tensor_list):
        pairs.append([line, tensor])
    return pairs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--input_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--pad_size', type=int, default=32)
    parser.add_argument('--train_path', default='./Weibo/data/train.csv', type=str)
    parser.add_argument('--dev_path', default='./Weibo/data/dev.csv', type=str)
    parser.add_argument('--test_path', default='./Weibo/data/test.csv', type=str)
    parser.add_argument('--vocab_path', default='./Weibo/data/vocab.pkl', type=str)
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    args = parser.parse_args()

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    embedding = load_pretrained_embeddings()
    embedding = torch.tensor(embedding, dtype=torch.float)

    print("Loading data...")
    vocab, train_dataset, dev_dataset, test_dataset = build_dataset(args, args.word)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    test_loader = DataLoader(dataset=train_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    model = BiLSTM(embedding, args.input_size, args.hidden_size, args.num_layers)

    model.eval()
    train_tensor = get_tensor(model, train_loader)
    dev_tensor = get_tensor(model, dev_loader)
    test_tensor = get_tensor(model, test_loader)

    train_text_tensor_pairs = get_text_tensor_pair(train_tensor, args.train_path)
    dev_text_tensor_pairs = get_text_tensor_pair(dev_tensor, args.dev_path)
    test_text_tensor_pairs = get_text_tensor_pair(test_tensor, args.test_path)

    # 存储数据
    import pickle
    pickle.dump(train_text_tensor_pairs, open('Weibo/tensor/train.pkl', 'wb'))
    pickle.dump(dev_text_tensor_pairs, open('Weibo/tensor/dev.pkl', 'wb'))
    pickle.dump(test_text_tensor_pairs, open('Weibo/tensor/test.pkl', 'wb'))

    # 读取数据
    data = pickle.load(open('Weibo/tensor/train.pkl', 'rb'))

    #连接milvus
    _HOST = '192.168.xx.xx'
    _PORT = 19530
    milvus = Milvus(_HOST, _PORT)
    collection_name = 'example_collection'
    partition_tag = 'demo_tag'
    segment_name= ''
    _DIM = 300
    ivf_param = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 8000}}
    milvus.create_index(collection_name, "embedding", ivf_param)
    vectors = [[random.random() for _ in range(_DIM)] for _ in range(8000)]
    ids = [i for i in range(8000)]
    collection_param = {
        "fields": [
            #  Milvus doesn't support string type now, but we are considering supporting it soon.
            #  {"name": "title", "type": DataType.STRING},
            {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 8000}},
        ],
        "segment_row_limit": 8000,
        "auto_id": False
    }
    embeddings = data
    hybrid_entities = [
        # Milvus doesn't support string type yet, so we cannot insert "title".
        {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR},
    ]

    for _ in range(8000):
        ids = milvus.insert(collection_name, hybrid_entities, ids, partition_tag="comment")
    print("\n----------insert----------")
    print("Films are inserted and the ids are: {}".format(ids))
    milvus.flush([collection_name])
    query_embedding = [random.random() for _ in range(300)]
    query_hybrid = {
        "bool": {
            "must": [
                {
                    "vector": {
                    "embedding": {"similar": 10, "query": [query_embedding], "metric_type": "L2"}
                    }
            }
        ]
      }
    }
    results = milvus.search(collection_name, query_hybrid, fields=["embedding"])
    for entities in results:
        for similar_comment in entities:
            current_entity = similar_comment.entity
            print("- id: {}".format(similar_comment.id))
            print("- embedding: {}".format(current_entity.embedding))


