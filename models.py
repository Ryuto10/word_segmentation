import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(self, n_vocab: int, embed_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout: float):
        super(Model, self).__init__()
        # hyper parameter
        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # parameter
        self.embed = nn.Embedding(self.n_vocab, self.embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, self.n_layers,
                            batch_first=True, dropout=self.dropout, bidirectional=True)
        self.linear = nn.Linear(self.hidden_dim * 2, self.out_dim)

    def forward(self, xs: (torch.Tensor, torch.Tensor)):
        # padded_xs: torch.Tensor (shape = (batch, max sentence))
        # xs_len: torch.Tensor (shape = batch)
        padded_xs, xs_len = xs

        # GPUへ転送
        if torch.cuda.is_available():
            padded_xs = padded_xs.cuda()
            xs_len = xs_len.cuda()

        # exs: torch.Tensor (shape = (batch, max sentence, embed))
        exs = self.embed(padded_xs)

        # out_lstm: torch.Tensor (shape = (batch, max sentence, hidden)
        packed_exs = pack_padded_sequence(exs, xs_len, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_exs)
        out_lstm, _ = pad_packed_sequence(packed_out, batch_first=True)

        # outs: torch.Tensor (shape = (batch, max sentence, 6)
        outs = self.linear(out_lstm)

        # score: [torch.Tensor, ...], length = batch size, each tensor shape = (sentence, 2)
        score1 = [self.extract_score(out, int(x_len), 0) for out, x_len in zip(outs, xs_len)]
        score2 = [self.extract_score(out, int(x_len), 1) for out, x_len in zip(outs, xs_len)]
        score3 = [self.extract_score(out, int(x_len), 2) for out, x_len in zip(outs, xs_len)]

        return score1, score2, score3

    @staticmethod
    def extract_score(out, x_len: int, kind: int):
        return out[:x_len, kind * 2:(kind + 1) * 2]


def evaluation(score1, score2, score3, ys):
    """
    Args:
        score1: [torch.Tensor (shape = (sentence, 2)), ...], length = batch
        score2: [torch.Tensor (shape = (sentence, 2)), , ...], length = batch
        score3: [torch.Tensor (shape = (sentence, 2)), , ...], length = batch
        ys: [torch.Tensor (shape = (sentence, 3)), , ...], length = batch
    Returns:
        accuracy1: The accuracy of whether the word starts
        accuracy2: The accuracy of whether it is a predicate
        accuracy3: The accuracy of whether the phrase begins
    """
    # shape = (batch * sentence, 2 or 3)
    cat_ys = torch.cat(ys, dim=0)
    cat_score1 = torch.cat(score1, dim=0).cpu()
    cat_score2 = torch.cat(score2, dim=0).cpu()
    cat_score3 = torch.cat(score3, dim=0).cpu()

    correct1 = sum(torch.argmax(cat_score1, dim=1) == cat_ys[:, 0])
    correct2 = sum(torch.argmax(cat_score2, dim=1) == cat_ys[:, 0])
    correct3 = sum(torch.argmax(cat_score3, dim=1) == cat_ys[:, 0])
    total = cat_ys.size(0)

    return correct1, correct2, correct3, total
