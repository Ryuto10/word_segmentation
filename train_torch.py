import argparse
import json
from os import path, mkdir

import torch
import torch.nn as nn
from scipy.stats.mstats import gmean
from tqdm import tqdm

from iterators import MyBucketIterator
from models import Model, evaluation


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--dataset', type=path.abspath,
                        help="Directory which contains dataset")
    parser.add_argument('--out_dir', default='result')
    parser.add_argument('--model', type=path.abspath, dest='model_path', default=None)

    # training option
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--early_stopping', type=int, default=None)
    parser.add_argument('--save_interval', type=int, default=None)

    # hyper parameters
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)

    return parser


def train(train_iter, test_iter, model, args):
    early_stopping_count = 0
    best_score = 0
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate,
                                 weight_decay=0.0001)

    for epoch in range(1, args.epoch + 1):
        print("# {} epoch".format(epoch))

        # Train
        print('## Train')
        model.train()
        total_loss = 0
        train_iter.create_batches()
        for xs, ys in tqdm(train_iter):
            # 勾配の初期化
            optimizer.zero_grad()
            model.zero_grad()

            # GPUへ転送
            if torch.cuda.is_available():
                ys = [y.cuda() for y in ys]

            # 学習
            scores = model(xs)
            loss = 0
            for idx, y in enumerate(ys):
                loss += loss_function(scores[idx].transpose(1, 2), y)
            loss.backward()
            total_loss += loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print("### Loss: {}".format(float(total_loss)))

        # Evaluation
        print("## Validation")
        model.eval()
        test_iter.create_batches()
        total_corr = torch.DoubleTensor([0, 0, 0])
        total_n = 0
        for xs, ys in tqdm(test_iter):
            scores = model(xs)
            corrects, total = evaluation(scores, ys)
            total_corr += corrects
            total_n += int(total)
        accuracy = (total_corr / total_n).tolist()
        eval_score = gmean(accuracy)
        print("### 単語ACC： {:.3f}, 述語ACC： {:.3f}, 文節ACC: {:.3f}, score: {:.3f}".format(*accuracy, eval_score))

        # Save best model
        if eval_score > best_score:
            best_score = eval_score
            early_stopping_count = 0
            torch.save(model, path.join(args.out_dir, "best.model"))
        else:
            early_stopping_count += 1

        # Early stopping
        if args.early_stopping and early_stopping_count > args.early_stopping:
            break

        # Save interval
        if args.save_interval and epoch % args.save_interval == 0:
            torch.save(model, path.join(args.out_dir, "{}epoch.model".format(epoch)))


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    if not path.exists(args.out_dir):
        mkdir(args.out_dir)
        print("# Make directory: {}".format(args.out_dir))

    torch.manual_seed(args.seed)

    with open(path.join(args.dataset, "train.json")) as fi:
        train_data = [json.loads(line) for line in fi]
    with open(path.join(args.dataset, "dev.json")) as fi:
        dev_data = [json.loads(line) for line in fi]
    with open(path.join(args.dataset, "char2index.json")) as fi:
        char2index = json.load(fi)

    train_iter = MyBucketIterator(train_data, args.batch, True)
    dev_iter = MyBucketIterator(dev_data, args.batch, False)

    model = Model(n_vocab=len(char2index),
                  embed_dim=args.embed_dim,
                  hidden_dim=args.hidden_dim,
                  out_dim=6,
                  n_layers=args.layers,
                  dropout=args.dropout)

    if args.model_path and path.exists(args.model_path):
        model = torch.load(args.model_path, map_location=torch.device('cpu'))

    if torch.cuda.is_available():
        model = model.cuda()

    train(train_iter, dev_iter, model, args)


if __name__ == '__main__':
    main()
