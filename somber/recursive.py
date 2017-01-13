import numpy as np
import logging
import time

from somber.som import Som
from somber.utils import expo, progressbar


logger = logging.getLogger(__name__)


class Recursive(Som):

    def __init__(self, map_dim, dim, learning_rate, alpha, beta, sigma=None, lrfunc=expo, nbfunc=expo):

        super().__init__(map_dim, dim, learning_rate, lrfunc, nbfunc, sigma)
        self.context_weights = np.random.uniform(0.0, 1.0, (self.map_dim, self.map_dim))
        self.alpha = alpha
        self.beta = beta

    def _train_loop(self, X, update_counter, context_mask):
        """
        The train loop. Is a separate function to accomodate easy inheritance.

        :param X: The input data.
        :param update_counter: A list of indices at which the params need to be updated.
        :return: None
        """

        epoch = 0
        influences = self._param_update(0, len(update_counter))

        prev_activation = np.zeros((self.map_dim,))

        for idx, x in enumerate(progressbar(X)):

            prev_activation = self._example(x, influences, prev_activation=prev_activation)

            if idx in update_counter:

                epoch += 1
                influences = self._param_update(epoch, len(update_counter))

            if idx in context_mask:

                prev_activation = np.zeros((self.map_dim,))

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

        # Invert activation so argmin can be used in downstream functions
        # more consistency.
        activation = 1 - np.exp(-(self.alpha * distance_x + self.beta * distance_y))

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

        prev_activation = np.zeros((self.map_dim,))

        for x in X:
            prev_activation, _, _ = self._get_bmus(x, prev_activation=prev_activation)
            distances.append(prev_activation)

        return distances