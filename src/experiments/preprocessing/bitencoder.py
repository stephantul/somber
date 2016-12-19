import numpy as np

from itertools import chain


class BitEncoder(object):

    def __init__(self):

        self.index = {}
        self.num_bits = None
        self.data = {}

    def fit_transform(self, data, maxlen=10):

        self.fit(data)
        return self.transform(data, maxlen)

    def fit(self, data):

        self.index = {item: idx for idx, item in enumerate(set(chain.from_iterable(data)))}
        self.num_bits = int(np.ceil(np.log2(max(self.index.values()))))

        for k, v in self.index.items():
            v = bin(v)[2:]
            d = [int(v[idx]) if idx < len(v) else 0 for idx in reversed(range(self.num_bits))]
            self.data[k] = d

    def transform(self, data, maxlen):

        X = []

        for seq in [seq for seq in data if len(seq) <= maxlen]:

            to_append = np.zeros((maxlen, self.num_bits))

            for idx, item in enumerate(seq):
                to_append[idx] += self.data[item]

            X.append(to_append)

        return np.array(X)

if __name__ == "__main__":

    p = ["Ik ben een hondje", "ik ben een kathe baha...!!!"]
    b = BitEncoder()
    z = b.fit_transform(p)

    print(z)
    print(b.data)
    print(b.index)
    print(b.num_bits)