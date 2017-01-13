import numpy as np
import logging
import time

from som import Som
from utils import expo, progressbar


logger = logging.getLogger(__name__)


class Recurrent(Som):

    def __init__(self, map_dim, dim, learning_rate, alpha, sigma=None, lrfunc=expo, nbfunc=expo):

        super().__init__(map_dim, dim, learning_rate, lrfunc, nbfunc, sigma)
        self.alpha = alpha

    def _train_loop(self, X, update_counter):
        """
        The train loop. Is a separate function to accomodate easy inheritance.

        :param X: The input data.
        :param update_counter: A list of indices at which the params need to be updated.
        :return: None
        """

        epoch = 0
        prev_activation = np.zeros((self.map_dim, self.data_dim))
        influences, learning_rates = self._param_update(0, len(update_counter))

        for idx, x in enumerate(progressbar(X)):

            prev_activation = self._example(x, influences, prev_activation=prev_activation)

            if idx in update_counter:

                epoch += 1
                influences, learning_rate = self._param_update(epoch, len(update_counter))

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

        # Get the indices of the Best Matching Units, given the data.
        activation, difference = self._get_bmus(x, prev_activation=prev_activation)

        influence, bmu = self._apply_influences(activation, influences)

        # Minibatch update of X and Y. Returns arrays of updates,
        # one for each example.
        self.weights += self._calculate_update(difference, influence)

        return difference

    def _get_bmus(self, x, **kwargs):
        """
        Gets the best matching units, based on euclidean distance.
        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        # Differences is the components of the weights subtracted from the weight vector.
        difference_x = self._distance_difference(x, self.weights)
        activation = (1 - self.alpha) * kwargs['prev_activation'] + (self.alpha * difference_x)

        # Distances are squared euclidean norm of differences.
        # Since euclidean norm is sqrt(sum(square(x)))) we can leave out the sqrt
        # and avoid doing an extra square.
        # Axis 2 because we are doing minibatches.
        distances = np.sum(np.square(activation), axis=1)

        return distances, activation

    def _predict_base(self, X):
        """
        Predicts distances to some input data.

        :param X: The input data.
        :return: An array of arrays, representing the activation
        each node has to each input.
        """

        # Return the indices of the BMU which matches the input data most
        distances = []

        prev_activation = np.zeros((self.map_dim, self.data_dim))

        for x in X:
            distance, prev_activation = self._get_bmus(x, prev_activation=prev_activation)
            distances.append(distance)

        return distances

