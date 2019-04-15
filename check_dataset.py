import json
import os.path
import argparse
from itertools import islice

def iter_char(xs, ys, index2char):
    in_prd = False
    for x, (f_word, f_section, f_prd) in zip(xs, ys):
        start_char = ""
        if not f_prd and in_prd:
            in_prd = False
            start_char = ")"
        if f_section:
            start_char += "/"
        elif f_word:
            start_char += " "
        if f_prd and not in_prd:
            in_prd = True
            start_char += "("

        yield start_char + index2char[x]

def get_visible_sent(xs, ys, index2char):
    return "".join(c for c in iter_char(xs, ys, index2char))

def main(args):
    with open(args.file) as f:
        data = json.load(f)
    with open(os.path.join(os.path.dirname(args.file), "char2index.json")) as f:
        char2index = json.load(f)
        index2char = {v : k for k, v in char2index.items()}

    for xs, ys in islice(data, args.start, args.end):
        print(get_visible_sent(xs, ys, index2char))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='makedata')
    parser.add_argument('file')
    parser.add_argument('--start', '-s', type=int, default=0)
    parser.add_argument('--end', '-e', type=int, default=10)
    main(parser.parse_args())
