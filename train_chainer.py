import argparse
import json
import os.path

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import training, reporter
from chainer.training import extensions


class MyModel(chainer.Chain):
    def __init__(self, embed_dim: int, h_dim: int, out_dim: int,
                 n_layers: int, n_vocab: int):
        super(MyModel, self).__init__()
        with self.init_scope():
            self.char_embed = L.EmbedID(n_vocab, embed_dim)
            self.lstm = L.NStepBiLSTM(n_layers=n_layers, in_size=embed_dim,
                                      out_size=h_dim, dropout=0.1)
            self.linear = L.Linear(in_size=h_dim * 2, out_size=out_dim)

    def forward(self, xs, ys):
        predict = self.prediction(xs)
        gold = F.concat(ys, axis=0)

        a1 = self.get_softmax_cross_entropy(predict, gold, 0)
        a2 = self.get_softmax_cross_entropy(predict, gold, 1)
        a3 = self.get_softmax_cross_entropy(predict, gold, 2)
        loss = a1 + a2 + a3
        reporter.report({'loss': loss}, self)

        return loss

    def get_softmax_cross_entropy(self, predict, gold, idx):
        return F.softmax_cross_entropy(predict[:, idx*2:(idx+1)*2], gold[:, idx],
                                       normalize=False)

    def prediction(self, xs):
        embed_xs = self.set_embed(xs)
        _, _, hs = self.lstm(None, None, embed_xs)
        concat_hs = F.concat(hs, axis=0)
        predict = self.linear(concat_hs)

        return predict

    def set_embed(self, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        conc_xs = F.concat(xs, axis=0)
        ex = self.char_embed(conc_xs)
        exs = F.split_axis(ex, x_section, 0)

        return exs


def convert(batch, device):
    sentences = [
        chainer.dataset.to_device(device, sentence) for sentence, _ in batch]
    labels = [chainer.dataset.to_device(device, label) for _, label in batch]

    return {'xs': sentences, 'ys': labels}


def main(args):
    with open(os.path.join(args.in_dir, "train.json")) as f:
        train_data = [(np.array(xs, np.int32), np.array(ys, np.int32)) for xs, ys in json.load(f)]
    with open(os.path.join(args.in_dir, "dev.json")) as f:
        dev_data = [(np.array(xs, np.int32), np.array(ys, np.int32)) for xs, ys in json.load(f)]
    with open(os.path.join(args.in_dir, 'char2index.json')) as f:
        char2index = json.load(f)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batch)
    dev_iter = chainer.iterators.SerialIterator(dev_data, args.batch,
                                                repeat=False, shuffle=False)
    model = MyModel(embed_dim=args.embed_dim,
                    h_dim=args.Hidden_dim,
                    out_dim=6,
                    n_layers=args.Layers,
                    n_vocab=len(char2index))

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)

    stop_trigger = training.triggers.EarlyStoppingTrigger(
        monitor='validation/main/loss', max_trigger=(args.epoch, 'epoch'),
        patients=5, mode='min')

    trainer = training.Trainer(updater, stop_trigger, out=args.out_dir)

    evaluator = extensions.Evaluator(
        dev_iter, model, device=args.gpu, converter=convert)

    trainer.extend(evaluator, trigger=(1, 'epoch'))

    trainer.extend(
        extensions.LogReport(trigger=(1, 'epoch'),
                             log_name='epoch_log'),
                             trigger=(1, 'epoch'), name='epoch')
    trainer.extend(
        extensions.LogReport(trigger=(10, 'iteration'),
                             log_name='iteration_log'),
                             trigger=(10, 'iteration'), name='iteration')

    entries = ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']
    trainer.extend(
        extensions.PrintReport(entries=entries, log_report='epoch'), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # 値のいいところでsaveする
    trainer.extend(
        extensions.snapshot_object(model, 'best_model.npz'),
        trigger=training.triggers.MinValueTrigger('validation/main/loss', (1, 'epoch')))

    print("start")
    trainer.run()
    print("end")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--batch', '-b', type=int, default=128)
    parser.add_argument('--epoch', '-e', type=int, default=150)
    parser.add_argument('--embed_dim', '-E', type=int, default=128)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--Hidden_dim', '-H', type=int, default=128)
    parser.add_argument('--in_dir', '-i', default="work/")
    parser.add_argument('--learning_rate', '-l', type=float, default=0.001)
    parser.add_argument('--Layers', '-L', type=int, default=2)
    parser.add_argument('--out_dir', '-o', default='result')
    main(parser.parse_args())
