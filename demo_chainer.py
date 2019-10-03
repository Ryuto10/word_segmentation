import argparse
import json

import chainer
import numpy as np
from chainer import serializers

from check_dataset import get_visible_sent
from train_chainer import MyModel

CHAR2INDEX = "work/char2index.json"
BEST_MODEL = "gpu_result/best_model.npz"
EMBED_DIM = 128
HIDDEN_DIM = 128
LAYERS = 2


def main(args):
    with open(CHAR2INDEX) as f:
        char2index = json.load(f)
        index2char = {v : k for k, v in char2index.items()}

    model = MyModel(embed_dim=EMBED_DIM,
                    h_dim=HIDDEN_DIM,
                    out_dim=6,
                    n_layers=LAYERS,
                    n_vocab=len(char2index))

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    serializers.load_npz(BEST_MODEL, model)

    while(1):
        text = input("text > ")
        if text == "q":
            break
        if text == "":
            continue
        x = np.array([char2index[c] if c in char2index else char2index["UNK"]
                      for c in text], np.int32)
        gpu_x = [chainer.dataset.to_device(args.gpu, x)]
        predict = model.prediction(gpu_x)
        predict = chainer.backends.cuda.to_cpu(predict.data)
        y1 = np.argmax(predict[:, 0:2], axis=1)
        y2 = np.argmax(predict[:, 2:4], axis=1)
        y3 = np.argmax(predict[:, 4:6], axis=1)
        y = [(i1, i2, i3) for i1, i2, i3 in zip(y1, y2, y3)]

        out = get_visible_sent(x.tolist(), y, index2char)
        print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    main(parser.parse_args())
