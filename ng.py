import numpy as np
# import tensorflow as tf
import time
import logging
import cProfile

from progressbar import progressbar
from collections import defaultdict

logger = logging.getLogger(__name__)


def expo(value, current_epoch, total_epochs):

    return value * np.exp(-current_epoch / total_epochs)

def static(value, current_epoch, total_epochs):

    return value


class Som(object):

    def __init__(self, map, dim, learning_rates, lrfunc=expo, nbfunc=expo):

        self.lam = 0

        if type(learning_rates) != list:
            learning_rates = [learning_rates]

        self.learning_rates = np.array(learning_rates)

        self.map_dim = map

        self.weights = np.random.uniform(0.0, 1.0, size=(self.map_dim, dim))
        self.data_dim = dim

        self.lrfunc = lrfunc
        self.trained = False

    def train(self, X, num_epochs=10, batch_size=100):
        """
        Fits the SOM to some data for a number of epochs.
        As the learning rate is decreased proportionally to the number
        of epochs, incrementally training a SOM is not feasible.

        :param X: the data on which to train.
        :param num_epochs: The number of epochs to simulate
        :return: None
        """

        # Scaler ensures that the neighborhood radius is 0 at the end of training
        # given a square map.
        # self.lam = num_epochs / np.log(self.sigma)

        # Local copy of learning rate.
        learning_rate = self.learning_rates

        bmus = []

        real_start = time.time()

        for epoch in range(num_epochs):

            print("\nEPOCH: {0}/{1}".format(epoch+1, num_epochs))
            start = time.time()

            # map_radius = self.nbfunc(self.sigma, epoch, self.lam)
            bmu = self.epoch_step(X, learning_rate[0], batch_size=batch_size)

            bmus.append(bmu)
            learning_rate = self.lrfunc(self.learning_rates, epoch, num_epochs)

            print("\nEPOCH TOOK {0:.2f} SECONDS.".format(time.time() - start))
            print("TOTAL: {0:.2f} SECONDS.".format(time.time() - real_start))

        self.trained = True

        return bmus

    def epoch_step(self, X, learning_rate, batch_size):
        """
        A single example.

        :param X: a numpy array of examples
        :param map_radius: The radius at the current epoch, given the learning rate and map size
        :param learning_rate: The learning rate.
        :param batch_size: The batch size to use.
        :return: The best matching unit
        """

        # Calc once per epoch
        # influences = self._distance_grid(map_radius) * learning_rate[0]
        # influences = np.asarray([influences] * self.data_dim).transpose((1, 2, 0))

        # One accumulator per epoch
        all_activations = []

        # Make a batch generator.
        accumulator = np.zeros_like(self.weights)
        num_updates = 0

        num_batches = np.ceil(len(X) / batch_size).astype(int)

        for index in progressbar(range(num_batches), idx_interval=1, mult=batch_size):

            # Select the current batch.
            batch = X[index * batch_size: (index+1) * batch_size]

            update, differences = self._batch(batch, learning_rate)

            all_activations.extend(np.sqrt(np.sum(np.square(differences), axis=2)))
            accumulator += update
            num_updates += 1

        self.weights += (accumulator / num_updates)

        return np.array(all_activations)

    def _batch(self, batch, alpha):

        distances, differences = self._get_bmus(batch)

        z = distances.argsort() / self.map_dim

        influences = np.exp(-z * z)
        influences = np.asarray([influences] * self.data_dim).transpose(1, 2, 0)

        update = differences * influences * alpha

        # influences = influences[bmus, :]
        # update = self._update(differences, influences).mean(axis=0)

        return update.mean(axis=0), differences

    def _update(self, input_vector, influence):
        """
        Updates the nodes, conditioned on the input vector,
        the influence, as calculated above, and the learning rate.

        :param input_vector: The input vector.
        :param influence: The influence the result has on each unit, depending on distance.
        """

        return input_vector * influence

    def _get_bmus(self, x):
        """
        Gets the best matching units, based on euclidean distance.

        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        differences = self._pseudo_distance(x, self.weights)
        distances = np.sqrt(np.sum(np.square(differences), axis=2))
        return distances, differences

    def _pseudo_distance(self, X, weights):
        """
        Calculates the euclidean distance between an input and all the weights in range.

        :param x: The input.
        :param weights: An array of weights.
        :return: The distance from the input of each weight.
        """

        # Correct
        p = np.tile(X, (1, self.map_dim)).reshape((X.shape[0], self.map_dim, X.shape[1]))
        return p - weights

    def predict(self, X):
        """
        Predicts node identity for input data.
        Similar to a clustering procedure.

        :param x: The input data.
        :return: A list of indices
        """

        # Return the indices of the BMU which matches the input data most
        bmus, _ = self._get_bmus(X)
        print(bmus.shape)
        return np.argmin(bmus, axis=1)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    colors = np.array(
         [[0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])

    colors = np.array(colors)

    # colors = []

    '''for x in range(10):
        for y in range(10):
            for z in range(10):
                colors.append((x/10, y/10, z/10))'''

    # colors = np.array(colors, dtype=float)
    # colors = np.vstack([colors, colors, colors, colors, colors, colors, colors, colors])

    '''addendum = np.arange(len(colors) * 10).reshape(len(colors) * 10, 1) / 10

    colors = np.array(colors)
    colors = np.repeat(colors, 10).reshape(colors.shape[0] * 10, colors.shape[1])

    print(colors.shape, addendum.shape)

    colors = np.hstack((colors,addendum))
    print(colors.shape)'''

    color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'greyblue', 'lilac', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'darkgrey', 'mediumgrey', 'lightgrey']

    s = Som(900, 3, [0.3, 0.3])
    start = time.time()
    bmus = s.train(colors, num_epochs=100, batch_size=1)

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))

    '''from visualization.umatrix import UMatrixView

    view = UMatrixView(500, 500, 'dom')
    view.create(s.weights, colors, s.width, s.height, bmus[-1])
    view.save("junk_viz/_{0}.svg".format(0))

    print("Made {0}".format(0))'''