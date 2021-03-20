import numpy as np
import sys


class Labels:
    def __init__(self, labels, dataset):
        self.labels = labels
        self.dataset = dataset
        self.l2i = {}
        self._index_dict()
        self.i2l = {}
        self._inverse_lookup()
        self._n_bytes = self.n_bytes()

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f"<Labels len='{self.__len__()}', n_bytes={self._n_bytes}>"

    def _inverse_lookup(self):
        self.i2l = {self.l2i[x]: x for x in self.l2i}

    def overwrite_lookup(self, l2i):
        self.l2i = l2i
        self._inverse_lookup()
        self._n_bytes = self.n_bytes()

    def n_bytes(self):
        val = sys.getsizeof(self.labels)+sys.getsizeof(self.dataset)+\
              sys.getsizeof(self.l2i)+sys.getsizeof(self.i2l)
        self._n_bytes = val
        return val

    def _index_dict(self):
        i = 0
        for label_list in self.labels:
            for label in label_list:
                if label not in self.l2i:
                    self.l2i[label] = i
                    i += 1

    def label_frequency(self):
        freq = np.zeros(len(self.l2i))
        for label in self.labels:
            if len(label) == 0:
                continue
            for l in label:
                l: str
                as_id = self.l2i[l]
                freq[as_id] = freq[as_id] + 1
        return freq
