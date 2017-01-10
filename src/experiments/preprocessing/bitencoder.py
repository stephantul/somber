import numpy as np

from itertools import chain


class BitEncoder(object):

    def __init__(self):

        self.index = {}
        self.num_bits = None
        self.data = {}

    def fit_transform(self, data):

        self.fit(data)
        return self.transform(data)

    def fit(self, data):

        self.index = {item: idx for idx, item in enumerate(set(chain.from_iterable(data)))}
        self.num_bits = int(np.ceil(np.log2(max(self.index.values()))))

        for k, v in self.index.items():
            v = bin(v)[2:]
            d = [int(v[idx]) if idx < len(v) else 0 for idx in reversed(range(self.num_bits))]
            self.data[k] = d

    def transform(self, sequence):

        transformed = np.zeros((len(sequence), self.num_bits))

        for idx, item in enumerate(sequence):
            transformed[idx] += self.data[item]

        return transformed
