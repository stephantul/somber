import numpy as np
from somber.pytorch.sequential import Recursive


def gen_data(dlen=10000):

    X = []
    desc = []

    while len(X) < dlen:

        gen = np.random.binomial(1, 0.4, 1)[0]
        if gen == 1:
            X.extend([[0,1,0], [0,0,1]])
            desc.extend([1, 2])
        else:
            X.append([1,0,0])
            desc.append(0)

    return np.array(X), desc


if __name__ == "__main__":

    import cProfile

    X, desc = gen_data(10000)
    # r = Recursive((10, 10), data_dim=3, learning_rate=.3, alpha=3.0, beta=.9)
    # cProfile.run("r.train(X, 2, show_progressbar=True, batch_size=1)")

    r_2 = Recursive((10, 10), data_dim=3, learning_rate=.3, alpha=3.0, beta=1.1)
    r_2.train(X, 2, show_progressbar=True, batch_size=2)