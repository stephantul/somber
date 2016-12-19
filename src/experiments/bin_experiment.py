import numpy as np
from recsom import RecSom

from src.experiments.markov_chain import MarkovGenerator
from utils import static

if __name__ == "__main__":

    m = MarkovGenerator(np.array([0, 1]), np.array([[0.7, 0.3], [0.4, 0.6]]), np.array([0.5, 0.5]))
    X = m.generate_sequences(120000, 1)

    print("Generated data")
    print(X.shape)

    # X = np.random.binomial(1, 0.3, size=(1000000,))

    m = RecSom(10, 10, 1, 0.3, sigma=10, alpha=1.25, beta=0.9, nbfunc=static, lrfunc=static)
    m.train(X, 100)