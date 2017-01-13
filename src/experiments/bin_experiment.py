import numpy as np
import logging

from merging import Merging
from recursive import Recursive
from som import Som
from src.experiments.markov_chain import MarkovGenerator
from utils import static, MultiPlexer, linear, expo

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # mgen = MarkovGenerator(np.array([[1, 0], [0, 1], [1, 1]]), np.array([[0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.1, 0.4, 0.5]]), np.array([0.2, 0.4, 0.4]))
    mgen = MarkovGenerator(np.array([[0, 1], [1, 0]]), np.array([[0.7, 0.3], [0.4, 0.6]]), np.array([1.0, 0.0]))
    X = mgen.generate_sequences(1, 300000)[0]

    # X = np.random.binomial(1, 0.5, size=(1000000,))

    print("Generated data")
    print(X.shape)

    # X = np.random.binomial(1, 0.3, size=(1000000,))

    r = Recursive((10, 10), 2, 0.1, alpha=2.0, beta=1.0, nbfunc=expo)
    r.train(MultiPlexer(X, 1), 1000, rough_epochs=1.0)