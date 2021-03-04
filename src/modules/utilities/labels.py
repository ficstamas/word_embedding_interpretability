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
