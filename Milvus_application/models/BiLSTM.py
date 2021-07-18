# coding: UTF-8
import torch
import torch.nn as nn
import pickle
from milvus import Milvus
from milvus import Status
from indexer.index import milvus_client, insert_vectors
import random
from milvus import DataType


_HOST = '192.168.xx.xx'
_PORT = 'xxx'
num = 6000


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'BiLSTM'
        self.train_path = dataset + '/data/train.csv'                                # 训练集
        self.dev_path = dataset + '/data/dev.csv'                                    # 验证集
        self.test_path = dataset + '/data/test.csv'                                  # 测试集
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(embedding) if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.class_list = ["negative", "positive"]                      # 类别
        self.num_classes = 2                                            # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 词向量维度
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, config, x):
        for i in range(config.Iterations):
            out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
            out, _ = self.lstm(out)
            with open("serialization", "wb") as f:
                pickle.dump(out, f)
        # out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

    milvus = Milvus(_HOST, _PORT)
    collection_name = 'example_collection_'
    partition_tag = 'demo_tag_'
    segment_name = ''
    # Generate a list of integer.
    list_of_int = [random.randint(0, 255) for _ in range(num)]
    # Generate 20 vectors of 128 dimension
    vectors = [[random.random() for _ in range(300)] for _ in range(num)]
    ids = [i for i in range(6000)]
    hybrid_entities = [
        {"name": "ID", "values": ids, "type": DataType.INT32},
        {"name": "Content", "values": vectors, "type": DataType.INT64},
        {"name": "Vec", "values": vectors, "type": DataType.FLOAT_VECTOR}
    ]
    # Insert vectors
    ids = milvus.insert('test01', hybrid_entities)
    with open("serialization", "wb") as f:
        res = pickle.load(f)
    for i in range(len(res)):
        ids =milvus.insert(collection_name, hybrid_entities, ids, partition_tag = "content")

    milvus.flush([collection_name])
    query_embedding = [random.random() for _ in range(8)]
    query_hybrid = {
        "bool": {
            "must": [
                {
                  "id": {"release_year": [2002, 2003]}
                },
                {
                    # "GT" for greater than
                    "range": {"duration": {"GT": 250}}
                },
                {
                "vector": {
                    "embedding": {"topk": 10, "query": [query_embedding], "metric_type": "L2"}
                }
            }
        ]
    }
    }

    # def forward(self, x, seq_len):
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)   # 变长序列
    #     out, _ = self.lstm(out)     # lstm输出维度: [batche_size, seq_len, num_directions * hidden_size]
    #     out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     # 句子最后时刻的hidden state通过linear层降维预测每个标签的概率: [batche_size, num_classes]
    #     out = self.fc(out[:, -1, :])
    #     return out
