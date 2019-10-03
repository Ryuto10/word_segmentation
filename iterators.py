import random
from math import ceil
from typing import List

import numpy as np
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence


class SameSentenceLengthBatchIterator(object):
    """Generate mini-batches with the same sentence length."""
    def __init__(self, dataset: List, batch_size: int = 128, sort_key=None):
        """
        Args:
            dataset: List of dataset.
            batch_size: Size of minibatch.
            sort_key: The function corresponding to the key of sort.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.sort_key = sort_key

        self._n_instances = len(self.dataset)
        self._n_batches = None
        self._order = None
        self._batch_order = None
        self._current_position = 0

        self._setting_order()
        self._setting_batch_order(self._order)

    def __len__(self):
        return self._n_batches

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_position == self._n_batches:
            self._current_position = 0
            raise StopIteration

        indices = self._batch_order[self._current_position]
        batch = [self.dataset[i] for i in indices]
        self._current_position += 1

        return self._batch_processor(batch)

    def shuffle(self):
        """Shuffle the order of the data."""
        order = [random.sample(indices, len(indices)) for indices in self._order]
        self._setting_batch_order(order)
        self._batch_order = random.sample(self._batch_order, len(self._batch_order))

    def _batch_processor(self, batch):
        """Change the minibatch size data to the format used by the model."""
        raise NotImplementedError

    def _setting_order(self):
        """Sort and set the order of dataset according to 'self.sort_key'."""
        sort_values = [self.sort_key(instance) for instance in self.dataset]
        self._order = []
        chunk = []
        sort_value = None

        for idx in np.argsort(sort_values):
            if chunk and sort_values[idx] != sort_value:
                self._order.append(chunk)
                chunk = []
            chunk.append(idx)
            sort_value = sort_values[idx]
        if chunk:
            self._order.append(chunk)

    def _setting_batch_order(self, order: List):
        """Split the dataset into minibatch sizes."""
        self._batch_order = []
        for indices in order:
            if len(indices) > self.batch_size:
                for i in range(ceil(len(indices) / self.batch_size)):
                    self._batch_order.append(indices[i * self.batch_size:(i + 1) * self.batch_size])
            else:
                self._batch_order.append(indices)
        self._n_batches = len(self._batch_order)
        assert sum(len(indices) for indices in self._batch_order) == self._n_instances


class PaddingBucketIterator(object):
    """Generate padded mini-batches to minimize padding as much as possible."""
    def __init__(self, dataset: List, sort_key=None, batch_size: int = 128,  shuffle=False, padding_value: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sort_key = sort_key
        self.padding_value = padding_value
        self.iterator = torchtext.data.BucketIterator(dataset,
                                                      batch_size=batch_size,
                                                      sort_key=sort_key,
                                                      shuffle=shuffle,
                                                      sort_within_batch=True)
        self.iterator.create_batches()

    def __len__(self):
        return ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        return self.padding(next(self.iterator.batches))

    def create_batches(self):
        self.iterator.create_batches()

    def padding(self, batch):
        """Return a padded mini-batch"""
        raise NotImplementedError


class MyBucketIterator(PaddingBucketIterator):
    """Generate padded mini-batches to minimize padding as much as possible about NAIST Text Corpus."""
    def __init__(self, dataset: List, batch_size=128, shuffle=False):
        super(MyBucketIterator, self).__init__(
            self.load_dataset(dataset), self.sort_key, batch_size, shuffle)

    @staticmethod
    def sort_key(instance):
        return len(instance[0])

    @staticmethod
    def load_dataset(dataset):
        loaded_dataset = []
        for sentence in dataset:
            x = torch.LongTensor(sentence[0])
            y = torch.LongTensor(sentence[1])
            loaded_dataset.append([x, y])

        return loaded_dataset

    def padding(self, batch):
        """
        Args:
            batch: [[xs, ys], ...], length = batch size
        Returns:
            padded_xs: torch.Tensor (shape = (batch, max sentence))
            xs_len: torch.Tensor (shape = batch)
            ys: [y, ...], length = batch size
                y: torch.Tensor (shape = (sentence, 3))
        """
        xs, ys = zip(*batch)
        xs_len = torch.Tensor([x.size(0) for x in xs])
        padded_xs = pad_sequence(xs, batch_first=True, padding_value=0)

        return [[padded_xs, xs_len], ys]
