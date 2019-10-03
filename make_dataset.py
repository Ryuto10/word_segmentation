import argparse
import glob
import json
from collections import defaultdict
from itertools import groupby, islice
from os import path, mkdir

from tqdm import tqdm

DATASET = ["train", "dev", "test"]


class Char2idx:
    def __init__(self):
        self.vocab = defaultdict(lambda: len(self.vocab))
        self.pad = self.vocab["<pad>"]
        self.UNK = self.vocab["<UNK>"]
        self.unk_counter = defaultdict(int)
        self.train = True

    def __call__(self, x):
        if self.train is False and x not in self.vocab:
            self.unk_counter[x] += 1
            return self.UNK
        else:
            return self.vocab[x]

    def display_vocab(self):
        print("# Number of char : {}".format(len(self.vocab)))
        print("# Number of UNK : {}".format(len(self.unk_counter)))

    def make_log(self, args, file):
        fo = open(file, "w")
        print("# Args :", file=fo)
        for k, v in args.__dict__.items():
            print("\t{} : {}".format(k, v), file=fo)
        print("# Number of char : {}".format(len(self.vocab)), file=fo)
        print("# Number of UNK : {}".format(len(self.unk_counter)), file=fo)
        print("# UNK char :", file=fo)
        for k, v in self.unk_counter.items():
            print("\t{} : {}".format(k, v), file=fo)
        fo.close()


class Doc:
    def __init__(self, file):
        with open(file) as fi:
            self.sents = [Sent(chunk) for v, chunk in groupby(fi, key=lambda x: x.startswith("EOS")) if not v]


class Sent:
    def __init__(self, chunk):
        self.sections = [Section(lines) for v, lines in groupby(chunk, key=lambda x: x.startswith("*")) if not v]


class Section:
    def __init__(self, lines):
        self.words = [Word(line) for line in lines]


class Word:
    def __init__(self, line):
        self.word = line.split("\t", 1)[0]
        self.is_pred = self.is_predicate(line)

    @staticmethod
    def is_predicate(line):
        if 'type="pred"' in line:
            return 1
        else:
            return 0


def create_instance(sent: Sent, char2idx: Char2idx) -> [[str, ...], [(int, int, int), ...]]:
    """
    Returns:
        xs: Sequence of character.
        ys: y[0] = Binary representing whether the word starts
            y[1] = Binary representing whether it is a predicate
            y[2] = Binary representing whether the phrase begins
    """
    xs = []
    ys = []
    for section in sent.sections:
        for idx_word, word in enumerate(section.words):
            for idx_char, c in enumerate(word.word):
                if word.is_pred and is_start(idx_char):
                    is_pred = 1
                else:
                    is_pred = 0

                if is_start(idx_word) and is_start(idx_char):
                    is_start_section = 1
                else:
                    is_start_section = 0

                xs.append(char2idx(c))
                ys.append((is_start(idx_char), is_pred, is_start_section))

    return [xs, ys]


def is_start(idx: int):
    if idx == 0:
        return 1
    else:
        return 0


def main(args):
    if not path.exists(args.out_dir):
        mkdir(args.out_dir)
        print('# make directory: "{}"'.format(args.out_dir))

    char2idx = Char2idx()

    for base_name in DATASET:
        if base_name == "train":
            char2idx.train = True
        else:
            char2idx.train = False

        in_files = sorted(glob.glob(path.join(args.in_dir, base_name, '*')))
        n_files = int(len(in_files) * args.ratio)
        print("# {} {}%".format(base_name, args.ratio * 100))

        # write
        with open(path.join(args.out_dir, base_name + ".json"), "w") as fo:
            for file in tqdm(islice(in_files, n_files)):
                doc = Doc(file)
                for sent in doc.sents:
                    pair = json.dumps(create_instance(sent, char2idx))
                    print(pair, file=fo)

    char2idx.display_vocab()
    char2idx.make_log(args, path.join(args.out_dir, "data_log.txt"))
    with open(path.join(args.out_dir, "char2index.json"), "w") as fo:
        json.dump(char2idx.vocab, fo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make dataset')
    parser.add_argument('--in_dir', '-i', type=path.abspath)
    parser.add_argument('--out_dir', '-o', type=path.abspath, default="./work")
    parser.add_argument('--ratio', '-r', type=float, default=1)
    main(parser.parse_args())
