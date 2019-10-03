import argparse
import json
from os import path, mkdir

import torch
import torch.nn as nn
from tqdm import tqdm

from iterators import MyBucketIterator
from models import evaluation
from check_dataset import get_visible_sent


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--dataset', type=path.abspath,
                        help="Directory which contains dataset")
    parser.add_argument('--out_dir', default='result')
    parser.add_argument('--model', type=path.abspath, dest='model_path', default=None)

    # option
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--eval', type=str, default=None,
                        help="Choose a data set from 'dev' or 'test' when testing")

    return parser


def test(test_data, model, char2index, args):
        if args.eval:
            # Evaluation
            print("## Evaluation: {}".format(args.eval))
            model.eval()
            total_corr1, total_corr2, total_corr3, total_n = 0, 0, 0, 0
            for xs, ys in tqdm(test_data):
                score1, score2, score3 = model(xs)
                correct1, correct2, correct3, total = evaluation(score1, score2, score3, ys)
                total_corr1 += int(correct1)
                total_corr2 += int(correct2)
                total_corr3 += int(correct3)
                total_n += int(total)
            acc1 = total_corr1 / total_n
            acc2 = total_corr2 / total_n
            acc3 = total_corr3 / total_n
            score = acc1 * acc2 * acc3
            print("### 単語ACC： {}, 述語ACC： {}, 文節ACC: {}, score: {}".format(acc1, acc2, acc3, score))

        if args.interactive:
            index2char = {idx: c for c, idx in char2index.items()}
            while 1:
                in_text = input("> ")
                if in_text == "q":
                    break
                import ipdb;
                ipdb.set_trace()
                x = [char2index[c] if c in char2index else char2index["<UNK>"] for c in in_text]
                x_len = torch.LongTensor([len(x)])
                xs = torch.LongTensor([x])
                score1, score2, score3 = model([xs, x_len])
                ys = torch.stack([torch.argmax(score1[0], dim=1),
                                  torch.argmax(score2[0], dim=1),
                                  torch.argmax(score3[0], dim=1)]).transpose(0, 1)
                in_xs = xs[0].cpu().tolist()
                in_ys = ys.cpu().tolist()
                sent = get_visible_sent(in_xs, in_ys, index2char)
                print(sent)


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    if not path.exists(args.out_dir):
        mkdir(args.out_dir)
        print("# Make directory: {}".format(args.out_dir))

    with open(path.join(args.dataset, "char2index.json")) as fi:
        char2index = json.load(fi)

    test_data = []
    if args.eval:
        with open(path.join(args.dataset, "{}.json".format(args.eval))) as fi:
            test_data = [json.loads(line) for line in fi]
            test_data = MyBucketIterator(test_data, 128, False)

    model: nn.Module = None
    if args.model_path and path.exists(args.model_path):
        model = torch.load(args.model_path, map_location=torch.device('cpu'))

    if torch.cuda.is_available():
        model = model.cuda()

    test(test_data, model, char2index, args)


if __name__ == '__main__':
    main()
