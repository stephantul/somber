import logging
import numpy as np

from merging import Merging
from som import Som
from utils import MultiPlexer

from src.experiments.markov_chain import MarkovGenerator

if __name__ == "__main__":

    np.random.seed(22)
    logging.basicConfig(level=logging.INFO)

    transition = np.array([[0.2, 0.8, 0.0, 0.0], [0.0, 0.2, 0.8, 0.0], [0.0, 0.0, 0.2, 0.8], [0.8, 0.0, 0.0, 0.2]])
    representations = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    start_probabilities = np.array([1.0, 0.0, 0.0, 0.0])

    m = MarkovGenerator(representations, transition, start_probabilities)
    p = m.generate_sequences(1, 100000)[0]

    w = np.random.uniform(0.0, 1.0, size=(4, 2))

    m = Merging(10, 10, 2, 0.3, 0.0, 0.5)
    m.train(MultiPlexer(p, 1), total_epochs=1000)

    s = Som(10, 10, 2, 0.1)
    s.train(p, total_epochs=100)
