import numpy as np
import logging
import time

from somber.som import Som
from somber.utils import expo, progressbar


logger = logging.getLogger(__name__)


class Recursive(Som):

    def __init__(self, map_dim, dim, learning_rate, alpha, beta, sigma=None, lrfunc=expo, nbfunc=expo):

        super().__init__(map_dim, dim, learning_rate, lrfunc, nbfunc, sigma, min_max=np.argmax)
        self.context_weights = np.zeros((self.map_dim, self.map_dim))
        self.alpha = alpha
        self.beta = beta

    def _train_loop(self, X, num_epochs, lr_update_counter, nb_update_counter, context_mask, show_progressbar):
        """
        The train loop. Is a separate function to accomodate easy inheritance.

        :param X: The input data.
        :param lr_update_counter: A list of indices at which the params need to be updated.
        :return: None
        """

        # Don't use asserts, these can be disabled
        if X.shape[-1] != self.data_dim:
            raise ValueError("Data dim does not match dimensionality of the data")

        nb_step = 0
        lr_step = 0

        # Calculate the influences for update 0.
        map_radius = self.nbfunc(self.sigma, 0, len(nb_update_counter))
        learning_rate = self.lrfunc(self.learning_rate, 0, len(lr_update_counter))
        influences = self._calculate_influence(map_radius) * learning_rate
        update = False

        idx = 0

        for epoch in range(num_epochs):

            prev_activation = np.zeros((self.map_dim,))

            for x in progressbar(X, use=show_progressbar):

                prev_activation = self._example(x, influences, prev_activation=prev_activation)

                if idx in nb_update_counter:
                    nb_step += 1

                    map_radius = self.nbfunc(self.sigma, nb_step, len(nb_update_counter))
                    logger.info("Updated map radius: {0}".format(map_radius))
                    update = True

                if idx in lr_update_counter:

                    lr_step += 1

                    learning_rate = self.lrfunc(self.learning_rate, lr_step, len(lr_update_counter))
                    logger.info("Updated learning rate: {0}".format(learning_rate))
                    update = True

                if update:

                    influences = self._calculate_influence(map_radius) * learning_rate
                    update = False

                idx += 1

    def _example(self, x, influences, **kwargs):
        """
        A single epoch.
        :param X: a numpy array of data
        :param map_radius: The radius at the current epoch, given the learning rate and map size
        :param learning_rates: The learning rate.
        :param batch_size: The batch size
        :return: The best matching unit
        """

        prev_activation = kwargs['prev_activation']

        activation, diff_x, diff_context = self._get_bmus(x, prev_activation=prev_activation)

        influence, bmu = self._apply_influences(activation, influences)

        # Update
        self.weights += self._calculate_update(diff_x, influence)
        self.context_weights += self._calculate_update(diff_context, influence[:, 0, np.newaxis])

        return activation

    def quant_error(self, X):
        """
        Calculates the quantization error by taking the minimum euclidean
        distance between the units and some input.

        :param X: Input data.
        :return: A vector of numbers, representing the quantization error
        for each data point.
        """

        dist = self._predict_base(X)
        return 1.0 - np.max(dist, axis=1)

    def _get_bmus(self, x, **kwargs):
        """
        Gets the best matching units, based on euclidean distance.
        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        prev_activation = kwargs['prev_activation']

        # Differences is the components of the weights subtracted from the weight vector.
        difference_x = self._distance_difference(x, self.weights)
        difference_y = self._distance_difference(prev_activation, self.context_weights)

        # Distances are squared euclidean norm of differences.
        # Since euclidean norm is sqrt(sum(square(x)))) we can leave out the sqrt
        # and avoid doing an extra square.
        distance_x = np.sum(np.square(difference_x), axis=1)
        distance_y = np.sum(np.square(difference_y), axis=1)

        activation = np.exp(-(self.alpha * distance_x) - (self.beta * distance_y))

        return activation, difference_x, difference_y

    def _predict_base(self, X):
        """
        Predicts distances to some input data.

        :param X: The input data.
        :return: An array of arrays, representing the activation
        each node has to each input.
        """

        # Return the indices of the BMU which matches the input data most
        distances = []

        prev_activation = np.sum(np.square(self._distance_difference(X[0], self.weights)), axis=1)
        distances.append(prev_activation)

        for x in X[1:]:
            prev_activation, _, _ = self._get_bmus(x, prev_activation=prev_activation)
            distances.append(prev_activation)

        return distances

    def generate(self, number_of_steps, starting_activation=None):

        if starting_activation is None:
            starting_activation = np.zeros((1, self.data_dim,))

        prev_activation = np.zeros((self.map_dim,))

        for x in starting_activation:
            prev_activation, _, _ = self._get_bmus(x, prev_activation=prev_activation)

        generated = []

        p = self.min_max(prev_activation)

        for x in range(number_of_steps):
            prev_activation, _, _ = self._get_bmus(self.weights[p], prev_activation=prev_activation)
            p = self.min_max(prev_activation)
            generated.append(p)

        return generated
