import argparse
import json
import os.path

import chainer
import numpy as np
from chainer import serializers
from tqdm import tqdm

from train_chainer import MyModel


def main(args):
    with open(os.path.join(args.in_dir, "test.json")) as f:
        test_data = [(np.array(xs, np.int32), np.array(ys, np.int32)) for xs, ys in json.load(f)]
        test_iter = chainer.iterators.SerialIterator(test_data, args.batch,
                                                     repeat=False, shuffle=False)
    with open(os.path.join(args.in_dir, 'char2index.json')) as f:
        char2index = json.load(f)

    model = MyModel(embed_dim=args.embed_dim,
                    h_dim=args.Hidden_dim,
                    out_dim=6,
                    n_layers=args.Layers,
                    n_vocab=len(char2index))

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    serializers.load_npz(os.path.join(args.out_dir, 'best_model.npz'), model)

    correct = 0
    total = 0
    for batch in tqdm(test_iter):
        xs = [chainer.dataset.to_device(args.gpu, sentence) for sentence, _ in batch]
        ys = [label for _, label in batch]
        predict = model.prediction(xs)
        predict = chainer.backends.cuda.to_cpu(predict.data)
        p1 = np.argmax(predict[:, 0:2], axis=1)
        p2 = np.argmax(predict[:, 2:4], axis=1)
        p3 = np.argmax(predict[:, 4:6], axis=1)
        predict = np.concatenate((p1.reshape(1, -1).T,
                                  p2.reshape(1, -1).T,
                                  p3.reshape(1, -1).T), axis=1)
        gold = np.concatenate(ys)
        correct += sum(predict == gold)
        total += len(gold)
    print("correct : {}".format(correct))
    print("total : {}".format(total))
    print("accuracy : {}".format(correct/total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--batch', '-b', type=int, default=128)
    parser.add_argument('--embed_dim', '-E', type=int, default=128)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--Hidden_dim', '-H', type=int, default=128)
    parser.add_argument('--in_dir', '-i', default="work/")
    parser.add_argument('--Layers', '-L', type=int, default=2)
    parser.add_argument('--out_dir', '-o', default='result')
    main(parser.parse_args())
