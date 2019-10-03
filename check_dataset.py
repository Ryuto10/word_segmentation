import argparse
import json
from os import path
from itertools import islice
from toolz import sliding_window


def get_visible_sent(xs, ys, index2char):
    """
    Args:
        xs: Sequence of character.
        ys: [y, ...]
            y[0] = Binary representing whether the word starts
            y[1] = Binary representing whether it is a predicate
            y[2] = Binary representing whether the phrase begins
        index2char: Converter (from index to char)
    """
    sent = []

    word_segment = [idx for idx, y in enumerate(ys) if y[0] == 1]
    preds = [ys[idx][1] for idx in word_segment]
    phrase_segment = [ys[idx][2] for idx in word_segment]

    indices = word_segment + [len(xs)]
    words = ["".join(index2char[x] for x in xs[ids[0]:ids[1]])
             for ids in sliding_window(2, indices)]

    for word, is_pred, phrase in zip(words, preds, phrase_segment):
        if is_pred:
            word = "(" + word + ")"
        if phrase:
            word = "/ " + word
        sent.append(word)

    return " ".join(sent)


def main(args):
    if args.vocab is None:
        vocab_fn = path.join(path.dirname(args.file), "char2index.json")
    else:
        vocab_fn = args.vocab

    with open(vocab_fn) as f:
        char2index = json.load(f)
        index2char = {v: k for k, v in char2index.items()}

    with open(args.file) as fi:
        for line in islice(fi, args.start, args.end):
            xs, ys = json.loads(line)
            print(get_visible_sent(xs, ys, index2char))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='makedata')
    parser.add_argument('file', type=path.abspath)
    parser.add_argument('--vocab', type=path.abspath)
    parser.add_argument('--start', '-s', type=int, default=0)
    parser.add_argument('--end', '-e', type=int, default=10)
    main(parser.parse_args())
