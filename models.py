import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    """This model infers word breaks, predicate identification, and phrase breaks for input sentences."""
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

    def forward(self, xs: (torch.Tensor, torch.Tensor)) -> [torch.Tensor, ...]:
        """
        Args:
            xs: Tuple of (padded_xs, xs_len)
                padded_xs: torch.Tensor, shape = (batch, max sentence)
                xs_len: torch.Tensor, shape = batch
        Return:
            scores: [score, ...], length = batch size
                score: torch.Tensor, shape = (sentence, 3, 2)
        """
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

        scores = [out[:int(x_len)].reshape(-1, 3, 2) for out, x_len in zip(outs, xs_len)]

        return scores


def evaluation(scores, ys):
    """
    Args:
        scores: [torch.Tensor (shape = (sentence, 3, 2)), ...], length = batch
        ys: [torch.Tensor (shape = (sentence, 3)), , ...], length = batch
    Returns:
        accuracy1: The accuracy of whether the word starts
        accuracy2: The accuracy of whether it is a predicate
        accuracy3: The accuracy of whether the phrase begins
    """
    cat_ys = torch.cat(ys, dim=0)  # shape = (batch * sentence, 3)
    cat_scores = torch.cat(scores, dim=0).cpu()  # shape = (batch * sentence, 3, 2)
    predicts = torch.argmax(cat_scores, dim=2)  # shape = (batch * sentence, 3)
    corrects = torch.sum(predicts == cat_ys, dim=0).double()  # shape = 3
    total = cat_ys.size(0)

    return corrects, total
