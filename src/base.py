import logging
import time
import numpy as np

from collections import defaultdict
from utils import progressbar, expo
from functools import reduce


logger = logging.getLogger(__name__)


class Base(object):

    def __init__(self,
                 map_dim,
                 dim,
                 learning_rate,
                 lrfunc=expo,
                 nbfunc=expo,
                 calculate_distance_grid=None,
                 calculate_influence=None,
                 apply_influences=None,
                 sigma=None):

        if sigma is not None:
            self.sigma = sigma
        else:
            # Add small constant to sigma to prevent divide by zero for maps of size 2.
            self.sigma = (max(map_dim) / 2.0) + 0.01

        self.lam = 0

        self.learning_rate = learning_rate
        self.map_dimensions = map_dim
        self.map_dim = reduce(np.multiply, map_dim, 1)
        self.weights = np.random.uniform(-0.1, 0.1, size=(self.map_dim, dim))
        self.data_dim = dim

        self.lrfunc = lrfunc
        self.nbfunc = nbfunc

        self._apply_influences = apply_influences
        self._calculate_distance_grid = calculate_distance_grid
        self._dist_grid = calculate_influence
        self.distance_grid = self._calculate_distance_grid()

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

        influences, learning_rates = self._param_update(0, num_effective_epochs)

        epoch_counter = X.shape[0] // num_effective_epochs
        epoch = 0
        start = time.time()

        for idx, x in enumerate(progressbar(X)):

            self._example(x, influences)

            if idx % epoch_counter == 0:

                epoch += 1

                influences, learning_rates = self._param_update(epoch, num_effective_epochs)

        self.trained = True
        logger.info("Number of training items: {0}".format(X.shape[0]))
        logger.info("Number of items per epoch: {0}".format(epoch_counter))
        logger.info("Total train time: {0}".format(time.time() - start))

    def _example(self, x, influences, **kwargs):
        """
        A single example.

        :param x: a single example
        :param influences: an array with influence values.
        :return: The best matching unit
        """

        activation, difference_x = self._get_bmus(x)

        influences, bmu = self._apply_influences(activation, influences)
        self.weights += self._calculate_update(difference_x, influences)

        return bmu

    def _apply_influences(self, distances, influences):

        raise NotImplementedError

    def _param_update(self, epoch, num_epochs):

        learning_rate = self.lrfunc(self.learning_rate, epoch, num_epochs)
        map_radius = self.nbfunc(self.sigma, epoch, self.lam)

        influences = self._calc_influence(map_radius) * learning_rate

        logging.info("RADIUS: {0}".format(map_radius))

        return influences, learning_rate

    def _calc_influence(self, sigma):
        """


        :param sigma:
        :return:
        """

        raise NotImplementedError

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

    def _init_distance_grid(self):
        """


        :return:
        """

        raise NotImplementedError

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

    def receptive_field(self, X):

        p = self.predict(X)

        prev = None
        fields = defaultdict(list)
        currfield = []

        for idx, ip in enumerate(zip(X, p)):

            item, prediction = ip

            if prediction == prev:
                currfield.append(X[idx-1])
            else:
                if currfield:
                    currfield.append(X[idx-1])
                    fields[p[idx-1]].append(currfield)
                currfield = []
            prev = prediction

        return {k: [z for z in v] for k, v in fields.items()}

    def invert_projection(self, X, identities):

        # Remove all duplicates from X
        X_unique, names = zip(*set([tuple((tuple(s), n)) for s, n in zip(X, identities)]))
        node_match = []

        for node in self.weights:

            differences = self._pseudo_distance(node, X_unique)
            distances = np.sqrt(np.sum(np.square(differences), axis=1))
            node_match.append(names[np.argmin(distances)])

        return node_match