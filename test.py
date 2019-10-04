import argparse
import json
from os import path

import torch
import torch.nn as nn
from scipy.stats.mstats import gmean
from tqdm import tqdm

from check_dataset import get_visible_sent
from iterators import MyBucketIterator
from models import evaluation


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--model', type=path.abspath, dest='model_path', default=None,
                        help="Path to model file.")

    # option
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--interactive', action='store_true')

    # evaluation option
    parser.add_argument('--test_file', type=path.abspath,
                        help="Path to test file.")

    # decode option
    parser.add_argument('--decode_file', type=path.abspath,
                        help="File with one sentence per line.")
    parser.add_argument('--out_file', type=path.abspath,
                        help="Output file.")

    # decode & interactive option
    parser.add_argument('--char2index', type=path.abspath,
                        help="Path to file of converter (character -> index).")

    return parser


def evaluate(model: nn.Module, test_iter):
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

    return accuracy, eval_score


def decode(model: nn.Module, char2index: dict, decode_data, fo):
    index2char = {idx: c for c, idx in char2index.items()}

    for sentence in tqdm(decode_data):
        x = [char2index[c] if c in char2index else char2index["<UNK>"] for c in sentence]

        xs = torch.LongTensor([x])
        x_len = torch.LongTensor([len(x)])

        scores = model([xs, x_len])

        ys = torch.argmax(scores[0], dim=2)

        # Write
        xs_list = xs[0].cpu().tolist()
        ys_list = ys.cpu().tolist()
        sent = get_visible_sent(xs_list, ys_list, index2char)
        print(sent, file=fo)


def interactive(model: nn.Module, char2index: dict):
    index2char = {idx: c for c, idx in char2index.items()}
    print("# Press 'q' to quit.")
    while 1:
        in_text = input("> ")
        if in_text == "q":
            break

        x = [char2index[c] if c in char2index else char2index["<UNK>"] for c in in_text]

        xs = torch.LongTensor([x])
        x_len = torch.LongTensor([len(x)])

        scores = model([xs, x_len])

        ys = torch.argmax(scores[0], dim=2)

        # Print
        xs_list = xs[0].cpu().tolist()
        ys_list = ys.cpu().tolist()
        sent = get_visible_sent(xs_list, ys_list, index2char)
        print(sent)


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    model: nn.Module = None
    if args.model_path and path.exists(args.model_path):
        model = torch.load(args.model_path, map_location=torch.device('cpu'))

    if torch.cuda.is_available():
        model = model.cuda()

    if args.eval:
        with open(args.test_file) as fi:
            test_data = [json.loads(line) for line in fi]
            test_iter = MyBucketIterator(test_data, 128, False)
        print("# Test: {}".format(args.test_file))
        accuracy, eval_score = evaluate(model, test_iter)
        print("# 単語ACC： {:.3f}, 述語ACC： {:.3f}, 文節ACC: {:.3f}, score: {}".format(*accuracy, eval_score))

    if args.decode:
        with open(args.char2index) as fi:
            char2index = json.load(fi)
        with open(args.decode_file) as fi:
            decode_data = [line.rstrip("\n") for line in fi]
        with open(args.out_file, "w") as fo:
            decode(model, char2index, decode_data, fo)

    if args.interactive:
        with open(args.char2index) as fi:
            char2index = json.load(fi)
        interactive(model, char2index)


if __name__ == '__main__':
    main()
