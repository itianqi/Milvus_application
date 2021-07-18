# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        # 根据训练集构建词表
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    # 句子最大长度为pad_size, 长减短补
    def load_dataset(path, pad_size=32):
        contents = []
        labels = []
        seq_lens = []
        with open(path, 'r', encoding='UTF-8') as f:
            for i, line in enumerate(f):
                lin = line.strip()
                if not lin or i == 0:
                    continue
                label, content = lin.split(',', 1)
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id: 根据句子中每个词找词表中对应的index
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append(words_line)
                labels.append(int(label))
                seq_lens.append(seq_len)
        return contents, labels, seq_lens
    train_input, train_label, train_seq = load_dataset(config.train_path, config.pad_size)
    dev_input, dev_label, dev_seq = load_dataset(config.dev_path, config.pad_size)
    test_input, test_label, test_seq = load_dataset(config.test_path, config.pad_size)
    train_dataset = create_dataset(train_input, train_label)
    dev_dataset = create_dataset(dev_input, dev_label)
    test_dataset = create_dataset(test_input, test_label)
    return vocab, train_dataset, dev_dataset, test_dataset


class create_dataset(Dataset):
    def __init__(self, inputs, labels):
        assert len(inputs) == len(labels)
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, item):
        return torch.LongTensor(self.inputs[item]),\
               torch.LongTensor([self.labels[item]])

    def __len__(self):
        return len(self.inputs)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def load_pretrained_embeddings():
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"   # 训练集
    vocab_dir = "./THUCNews/data/vocab.pkl"   # 词表
    pretrain_dir = "./THUCNews/data/sgns.weibo.bigram-char"    # 预训练词向量文件
    emb_dim = 300
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    # 随机初始化embeddings
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    # 若词表中的词在预训练词向量中, 则使用该词的预训练embedding覆盖原初始化的embeddings
    for i, line in enumerate(f.readlines()):
        if i == 0:  # 若第一行是标题，则跳过
            continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    return embeddings
