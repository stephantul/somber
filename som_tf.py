import numpy as np
import tensorflow as tf
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

    def __init__(self, width, height, dim, alpha):

        # Normal components
        self.scaler = 0

        self.alpha = alpha
        self.sigma = max(width, height) / 2

        self.width = width
        self.height = height

        self.trained = False
        self.data_dim = dim
        self.map_dim = width * height

        self._graph = tf.Graph()

    def _initialize_graph(self, batch_size, num_epochs):

        with self._graph.as_default():
            self.weights = tf.Variable(tf.random_normal([self.map_dim, self.data_dim]))
            self.vect_input = tf.placeholder("float", [batch_size, self.data_dim])
            self.epoch = tf.placeholder("float")
            self.distance_grid = tf.constant(self._calculate_distance_grid())

            tiles = tf.tile(self.vect_input, (1, self.map_dim))
            reshaped = tf.reshape(tiles, (batch_size, self.map_dim, self.data_dim))
            spatial_activation = tf.sub(reshaped, self.weights)

            euclidean = tf.sqrt(tf.reduce_sum(tf.square(spatial_activation), 2))
            bmus = tf.argmin(euclidean, 1)

            _learning_rate_op = tf.exp(tf.div(-self.epoch, num_epochs))
            _alpha_op = tf.mul(self.alpha, _learning_rate_op)
            _sigma_op = tf.mul(self.sigma, _learning_rate_op)

            neighborhood_func = tf.exp(tf.div(tf.cast(
                self.distance_grid, "float32"), tf.square(tf.mul(2.0, _sigma_op))))

            neighbourhood_func = tf.mul(_alpha_op, neighborhood_func)

            influences = tf.gather(neighbourhood_func, bmus)

            influences = tf.tile(influences, (1, self.data_dim))
            influences = tf.transpose(tf.reshape(influences, (batch_size, self.data_dim, self.map_dim)), (0, 2, 1))

            spatial_delta = tf.reduce_mean(tf.mul(influences, spatial_activation), 0)

            new_weights = tf.add(self.weights,
                                 spatial_delta)
            self._training_op = tf.assign(self.weights,
                                          new_weights)

            self._sess = tf.Session()
            self._sess.run(tf.initialize_all_variables())

    def train(self, X, num_epochs=10, batch_size=100):
        """
        Fits the SOM to some data for a number of epochs.
        As the learning rate is decreased proportionally to the number
        of epochs, incrementally training a SOM is not feasible.

        :param X: the data on which to train.
        :param num_epochs: The number of epochs to simulate
        :return: None
        """

        self._initialize_graph(batch_size, num_epochs)

        bmus = []

        real_start = time.time()

        closest = int(np.ceil(X.shape[0] / batch_size))
        print("X has {0} instances".format(X.shape[0]))

        shape = list(X.shape)
        shape[0] = closest * batch_size
        X = np.resize(X, shape)
        print("X was reshaped to {0} instances".format(X.shape[0]))

        for epoch in range(num_epochs):

            start = time.time()

            self.epoch_step(X, epoch, batch_size)
            print("EPOCH {0}/{1} TOOK {2} seconds".format(epoch, num_epochs, time.time() - start))
            print("TOTAL TIME {0}".format(time.time() - real_start))

        self.trained = True

        return bmus

    def epoch_step(self, X, epoch, batch_size):

        num_batches = np.ceil(len(X) / batch_size).astype(int)

        for index in progressbar(range(num_batches), idx_interval=1, mult=batch_size):
            batch = X[index * batch_size: (index + 1) * batch_size]

            self._sess.run(self._training_op,
                           feed_dict={self.vect_input: batch,
                                      self.epoch: epoch})

    def _distance_grid(self, radius):

        p = tf.exp(-1.0 * self.distance_grid / (2.0 * radius ** 2)).reshape(self.map_dim, self.map_dim)
        p = tf.repeat(self.data_dim).reshape(self.map_dim, self.map_dim, self.data_dim)

        return p

    def _batch(self, batch, influences):

        bmus, differences = self._get_bmus(batch)

        influences = influences[bmus]
        update = self._update(differences, influences).mean(axis=0)

        return update, differences

    def _update(self, input_vector, influence):
        """
        Updates the nodes, conditioned on the input vector,
        the influence, as calculated above, and the learning rate.

        :param input_vector: The input vector.
        :param influence: The influence the result has on each unit, depending on distance.
        """

        return influence * input_vector

    def _get_bmus(self, x):
        """
        Gets the best matching units, based on euclidean distance.

        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        differences = self._pseudo_distance(x, self.weights)
        distances = np.sqrt(np.sum(np.square(differences), axis=2))
        return np.argmin(distances, axis=1), differences

    def _pseudo_distance(self, X, weights):
        """
        Calculates the euclidean distance between an input and all the weights in range.

        :param x: The input.
        :param weights: An array of weights.
        :return: The distance from the input of each weight.
        """
        p = np.tile(X, (1, self.map_dim)).reshape((X.shape[0], self.map_dim, X.shape[1]))
        return p - weights

    def _calculate_distance_grid(self):

        distance_matrix = np.zeros((self.map_dim, self.map_dim))

        for i in range(self.map_dim):

            distance_matrix[i] = self._grid_dist(i).reshape(1, self.map_dim)

        return -1 * distance_matrix

    def _grid_dist(self, index):

        rows = self.height
        cols = self.width

        # bmu should be an integer between 0 to no_nodes
        node_col = int(index % cols)
        node_row = int(index / cols)

        r = np.arange(0, rows, 1)[:, np.newaxis]
        c = np.arange(0, cols, 1)
        dist2 = (r-node_row)**2 + (c-node_col)**2

        return dist2.ravel()

    def predict(self, X):
        """
        Predicts node identity for input data.
        Similar to a clustering procedure.

        :param x: The input data.
        :return: A list of indices
        """

        # Return the indices of the BMU which matches the input data most
        bmus, _ = self._get_bmus(X)
        return bmus

    def predict_distance(self, X):

        _, differences = self._get_bmus(X)
        return np.sqrt(np.sum(np.square(differences), axis=2))

    def map_weights(self):
        """
        Retrieves the grid as a list of lists of weights. For easy visualization.

        :return: A three-dimensional Numpy array of values (width, height, data_dim)
        """

        mapped_weights = []

        with self._sess.as_default():

            weights = self.weights.eval()

            for x in range(self.width):
                x *= self.height
                temp = []
                for y in range(self.height):
                    temp.append(weights[x + y])

                mapped_weights.append(temp)

        return np.array(mapped_weights)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    '''colors = np.array(
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

    colors = np.array(colors)'''

    colors = []

    for x in range(10):
        for y in range(10):
            for z in range(10):
                colors.append((x/10, y/10, z/10))

    colors = np.array(colors, dtype=float)

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

    s = Som(20, 20, 3, 1.0)
    start = time.time()
    bmus = s.train(colors, batch_size=100, num_epochs=100)

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))

    '''from visualization.umatrix import UMatrixView

    view = UMatrixView(500, 500, 'dom')
    view.create(s.weights, colors, s.width, s.height, bmus[-1])
    view.save("junk_viz/_{0}.svg".format(0))

    print("Made {0}".format(0))'''