import numpy as np


class Labels:
    def __init__(self, labels, dataset):
        self.labels = labels
        self.dataset = dataset
        self.l2i = {}
        self._index_dict()
        self.i2l = {self.l2i[x]: x for x in self.l2i}

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
