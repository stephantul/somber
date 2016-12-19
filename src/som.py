import logging
import time
import numpy as np

from collections import defaultdict
from utils import MultiPlexer, progressbar, expo


class Som(object):

    def __init__(self, width, height, dim, learning_rate, lrfunc=expo, nbfunc=expo, sigma=None):

        if sigma is not None:
            self.sigma = sigma
        else:
            # Add small constant to sigma to prevent divide by zero for maps of size 2.
            self.sigma = (max(width, height) / 2.0) + 0.01

        self.lam = 0

        self.learning_rate = learning_rate

        self.width = width
        self.height = height
        self.map_dim = width * height

        self.weights = np.random.uniform(-0.1, 0.1, size=(self.map_dim, dim))
        self.data_dim = dim

        self.distance_grid = self._calculate_distance_grid()

        self._index_dict = {idx: (idx // self.height, idx % self.height) for idx in range(self.weights.shape[0])}
        self._coord_dict = defaultdict(dict)

        self.lrfunc = lrfunc
        self.nbfunc = nbfunc

        for k, v in self._index_dict.items():

            x_, v_ = v
            self._coord_dict[x_][v_] = k

        self.trained = False

    def train(self, X, num_effective_epochs=10):
        """
        Fits the SOM to some data for a number of epochs.
        As the learning rate is decreased proportionally to the number
        of epochs, incrementally training a SOM is not feasible.

        :param X: the data on which to train.
        :param num_effective_epochs: The number of epochs to simulate
        :return: None
        """

        # Scaler ensures that the neighborhood radius is 0 at the end of training
        # given a square map.
        self.lam = num_effective_epochs / np.log(self.sigma)

        # Local copy of learning rate.
        influences, learning_rates = self._param_update(0, num_effective_epochs)

        epoch_counter = X.shape[0] / num_effective_epochs
        epoch = 0
        start = time.time()

        for idx, x in enumerate(progressbar(X)):

            self._example(x, influences)

            if idx % epoch_counter == 0:

                epoch += 1

                influences, learning_rates = self._param_update(epoch, num_effective_epochs)

                print("\nEPOCH TOOK {0:.2f} SECONDS.".format(time.time() - start))

        self.trained = True
        print("Number of training items: {0}".format(X.shape[0]))
        print("Number of items per epoch: {0}".format(epoch_counter))

    def _example(self, x, influences, **kwargs):
        """
        A single example.

        :param x: a single example
        :param influences: an array with influence values.
        :return: The best matching unit
        """

        distances, differences = self._get_bmus(x)

        bmu = np.argmin(distances)

        influences = influences[bmu]
        update = self._calculate_update(differences, influences)
        self.weights += update

        return bmu

    def _param_update(self, epoch, num_epochs):

        learning_rate = self.lrfunc(self.learning_rate, epoch, num_epochs)
        map_radius = self.nbfunc(self.sigma, epoch, self.lam)

        influences = self._calc_influence(map_radius) * learning_rate
        influences = np.asarray([influences] * self.data_dim).transpose((1, 2, 0))

        print("\nRADIUS: {0}".format(map_radius))

        return influences, learning_rate

    def _calc_influence(self, radius):
        """


        :param radius:
        :return:
        """

        p = np.exp(-1.0 * self.distance_grid / (2.0 * radius ** 2)).reshape(self.map_dim, self.map_dim)

        return p

    def _calculate_update(self, input_vector, influence):
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
        distances = np.sqrt(np.sum(np.square(differences), axis=1))
        return distances, differences

    def _pseudo_distance(self, x, weights):
        """
        Calculates the euclidean distance between an input and all the weights in range.

        :param x: The input.
        :param weights: An array of weights.
        :return: The distance from the input of each weight.
        """
        return x - weights

    def _calculate_distance_grid(self):
        """


        :return:
        """

        distance_matrix = np.zeros((self.map_dim, self.map_dim))

        for i in range(self.map_dim):

            distance_matrix[i] = self._grid_dist(i).reshape(1, self.map_dim)

        return distance_matrix

    def _grid_dist(self, index):
        """


        :param index:
        :return:
        """

        rows = self.height
        cols = self.width

        # bmu should be an integer between 0 to no_nodes
        node_col = int(index % cols)
        node_row = int(index / cols)

        r = np.arange(0, rows, 1)[:, np.newaxis]
        c = np.arange(0, cols, 1)
        dist2 = (r-node_row)**2 + (c-node_col)**2

        return dist2.ravel()

    def _predict_base(self, X):
        """
        Predicts node identity for input data.
        Similar to a clustering procedure.

        :param x: The input data.
        :return: A list of indices
        """

        # Return the indices of the BMU which matches the input data most
        distances = []

        for x in X:
            distance, _ = self._get_bmus(x)
            distances.append(distance)

        return distances

    def quant_error(self, X):
        """
        :param X:
        :return:
        """

        dist = self._predict_base(X)
        return np.min(dist, axis=1)

    def predict(self, X):

        dist = self._predict_base(X)
        return np.argmin(dist, axis=1)

    def map_weights(self):
        """
        Retrieves the grid as a list of lists of weights. For easy visualization.

        :return: A three-dimensional Numpy array of values (width, height, data_dim)
        """

        return self.weights.reshape((self.width, self.height, self.data_dim))

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    colors_ = np.array(
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

    colors = []

    for x in range(10):
        for y in range(10):
            for z in range(10):
                colors.append((x/10, y/10, z/10))

    colors_ = np.array(colors, dtype=float)

    colors = MultiPlexer(colors_, 10)

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

    s = Som(20, 20, 3, 0.3, sigma=10)
    start = time.time()
    bmus = s.train(colors, num_effective_epochs=100)

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))

    '''from visualization.umatrix import UMatrixView

    view = UMatrixView(500, 500, 'dom')
    view.create(s.weights, colors, s.width, s.height, bmus[-1])
    view.save("junk_viz/_{0}.svg".format(0))

    print("Made {0}".format(0))'''