import numpy as np
from msom_no_batch import MSom

if __name__ == "__main__":

    X = np.random.binomial(1, 0.3, size=(1000000,))

    m = MSom(12, 12, 1, [0.03], alpha=0.001, beta=0.5)
    m.train(X, 3)