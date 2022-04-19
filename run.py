# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network, train_free, test
from importlib import import_module
from adversary import FGM, FGSM, PGD, Free
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')

parser.add_argument('--adv_mode', default='baseline', type=str, help='Adversary mode, FGSM, PGD, Free or Baseline')

args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样



    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


    # train
    config.n_vocab = len(vocab)
    if args.adv_mode != 'Baseline':
        model = x.Model(config, need_dropout=False).to(config.device)
    else:
        model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)

    adv_model = None
    need_dropout = True

    config.adv_mode = args.adv_mode
    test(config, model, test_iter, need_dropout=False)

    if args.adv_mode == 'Free':
        adv_model = Free(model)
        config.learning_rate = 1e-4
        train_free(config, model, train_iter, dev_iter, test_iter, adv_model)
    else:
        if args.adv_mode == 'FGSM':
            adv_model = FGSM(model)
            need_dropout = False
        elif args.adv_mode == 'FGM':
            adv_model = FGM(model)
            need_dropout = False
        elif args.adv_mode == 'PGD':
            adv_model = PGD(model)
            need_dropout = False
        train(config, model, train_iter, dev_iter, test_iter, adv_model, need_dropout=need_dropout)



