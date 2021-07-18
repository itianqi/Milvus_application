# coding: UTF-8
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, get_time_dif, load_pretrained_embeddings

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, default='CNN', help='choose a model: BiLSTM, CNN')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'Weibo'  # 数据集

    if args.embedding == 'random':
        embedding = 'random'
    else:
        embedding = load_pretrained_embeddings()

    model_name = args.model

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    # 设置种子, 保证每次结果一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    vocab, train_dataset, dev_dataset, test_dataset = build_dataset(config, args.word)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=config.batch_size,
                            shuffle=False)
    test_loader = DataLoader(dataset=train_dataset,
                             batch_size=config.batch_size,
                             shuffle=False)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    # model = x.Model(config).to(config.device)
    model = x.Model(config)
    init_network(model)
    print(model.parameters)
    train(config, model, train_loader, dev_loader, test_loader)
