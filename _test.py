import numpy as np
import cProfile
import logging

from somber.merging import Merging
from somber.som import cosine
from somber.batch.som import Som

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    X = np.random.binomial(1, 0.5, 10000)[:, np.newaxis]


    def batch_distance(x, weights):
        """
        batched version of the euclidean distance.

        :param x: The input
        :param weights: The weights
        :return: A matrix containing the distance between each
        weight and each input.
        """

        m_norm = np.square(x).sum(axis=1)
        w_norm = np.square(weights).sum(axis=1)[:, np.newaxis]
        dotted = np.dot(np.multiply(x, 2), weights.T)

        res = np.outer(m_norm, np.ones((1, w_norm.shape[0])))
        res += np.outer(np.ones((m_norm.shape[0], 1)), w_norm.T)
        res -= dotted

        return res


    def batch_2(x, weights):

        d = np.dot(weights, x.T)
        d *= -2
        d += (np.sum(np.square(weights), axis=1)).reshape(len(weights), 1)
        return d.T

    X = np.array([X, X]).reshape(10000, 2)

    s = Som((20, 20), 2, 1.0, distance_function=batch_distance)
    s.trained = True
    s_2 = Som((20, 20), 2, 1.0, distance_function=batch_2)
    s_2.trained = True

    s.train(X, 50)
    s_2.train(X, 50)

    # r_2 = Merging((10, 10), 1, 1.0, 0.1, 0.5, distance_function=cosine)
    # cProfile.run("r_2.train(X, 100)")