import re
import glob
import json
import os.path
import argparse
import subprocess
from tqdm import tqdm
from collections import defaultdict
from itertools import groupby, islice

DATASET = "/Users/ryuto/lab/research/predicate_argument/data/train-with-juman"
BASENAMES = ["train", "dev", "test"]

class Obj2idx:
    def __init__(self):
        self.char2index = defaultdict(lambda: len(self.char2index))
        self.UNK = self.char2index["UNK"]
        self.counter = defaultdict(int)

    def __call__(self, x, unk=False):
        if unk and not x in self.char2index:
            self.counter[x] += 1
            return self.UNK
        else:
            return self.char2index[x]

    def show(self):
        print("Number of char : {}".format(len(self.char2index)))
        print("Number of UNK : {}".format(len(self.counter)))

    def make_log(self, args, file):
        fo = open(file, "w")
        print("args :", file=fo)
        for k, v in args.__dict__.items():
            print("\t{} : {}".format(k, v), file=fo)
        print("Number of char : {}".format(len(self.char2index)), file=fo)
        print("Number of UNK : {}".format(len(self.counter)), file=fo)
        print("UNK char :", file=fo)
        for k, v in self.counter.items():
            print("\t{} : {}".format(k, v), file=fo)
        fo.close()

class Doc:
    def __init__(self, file):
        self.sents = []
        with open(file) as f:
            for val, group in groupby(f, key=lambda x: x.startswith("EOS")):
                if not val:
                    self.sents.append(Sent(group))

class Sent:
    def __init__(self, chunk):
        self.sections = []
        for val, group in groupby(chunk, key=lambda x: x.startswith("*")):
            if not val:
                self.sections.append(Section(group))

    def get_xy(self):
        for section in self.sections:
            for word in section.words:
                for x, y in zip(word.xs, word.ys):
                    yield x, y

class Section:
    def __init__(self, chunk):
        self.words = []
        for idx, line in enumerate(chunk):
            self.words.append(Word(idx, line))

class Word:
    """
    x = (char)
    y = (word, section, pred)
    """
    def __init__(self, idx, line):
        self.xs = [x for x in line.split()[0]]
        self.ys = [y for y in self.iter_y(idx, line)]

    def iter_y(self, idx, line):
        word = line.split()[0]
        word_f = [0 for _ in range(len(word))]
        word_f[0] = 1
        section_f = [0 for _ in range(len(word))]
        if idx == 0:
            section_f[0] = 1
        if self.is_predicate(line):
            prd_f = [1 for _ in range(len(word))]
        else:
            prd_f = [0 for _ in range(len(word))]
        for y in zip(word_f, section_f, prd_f):
            yield y

    def is_predicate(self, line):
        if 'type="pred"' in line:
            return True
        else:
            return False

def main(args):
    if not os.path.exists(args.out_dir):
        subprocess.check_output( "mkdir {}".format(args.out_dir), shell=True )
        print('# create "{}"'.format(args.out_dir))

    obj2idx = Obj2idx()
    for basename in BASENAMES:
        if basename == "train":
            unk = False
        else:
            unk = True
        files = sorted(glob.glob(os.path.join(args.in_dir, basename, '*')))
        n_files = int(len(files) * args.ratio * 0.1)
        print("# {} {}%".format(basename, args.ratio * 10))

        data = []
        for file in tqdm(islice(files, n_files)):
            doc = Doc(file)
            for sent in doc.sents:
                xs, ys = [], []
                for x, y in sent.get_xy():
                    xs.append(obj2idx(x, unk))
                    ys.append(y)
                data.append((xs, ys))

        with open(os.path.join(args.out_dir, basename + ".json"), "w") as fo:
            json.dump(data, fo)

    obj2idx.show()
    obj2idx.make_log(args, os.path.join(args.out_dir, "data_log.txt"))
    with open(os.path.join(args.out_dir, "char2index.json"), "w") as fo:
        json.dump(obj2idx.char2index, fo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='makedata')
    parser.add_argument('--in_dir', '-i', default=DATASET)
    parser.add_argument('--out_dir', '-o', default="./work")
    parser.add_argument('--ratio', '-r', type=int, default=10)
    main(parser.parse_args())
