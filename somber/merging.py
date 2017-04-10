import numpy as np
import time
import logging

from somber.som import Som
from somber.utils import expo, progressbar
from collections import Counter


logger = logging.getLogger(__name__)


class Merging(Som):

    def __init__(self, map_dim, dim, learning_rate, alpha, beta, sigma=None, lrfunc=expo, nbfunc=expo):

        super().__init__(map_dim, dim, learning_rate, lrfunc, nbfunc, sigma=sigma)

        self.alpha = alpha
        self.beta = beta
        self.context_weights = np.ones(self.weights.shape)
        self.entropy = 0

    def _train_loop(self, X, num_epochs, lr_update_counter, nb_update_counter, context_mask, show_progressbar):

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
        print(nb_update_counter)
        print(lr_update_counter)

        bmu = None

        bmus = []
        prev_update = 0

        idx = 0

        for epoch in progressbar(range(num_epochs), use=show_progressbar):

            for x in X:

                bmu = self._example(x, influences, prev_bmu=bmu)
                bmus.append(bmu)

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
                    prev_update = self._entropy(Counter(bmus), prev_update)
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

        prev_bmu = kwargs['prev_bmu']

        if prev_bmu is None:
            context = np.zeros((self.data_dim,))
        else:
            context = (1 - self.beta) * self.weights[prev_bmu] + self.beta * self.context_weights[prev_bmu]

        # Get the indices of the Best Matching Units, given the data.
        activation, diff_x, diff_context = self._get_bmus(x, context=context)

        influence, bmu = self._apply_influences(activation, influences)

        # Minibatch update of X and Y. Returns arrays of updates,
        # one for each example.
        self.weights += self._calculate_update(diff_x, influence)
        self.context_weights += self._calculate_update(diff_context, influence)

        return bmu

    def _entropy(self, prev_bmus, prev_update):
        """
        Calculates the entropy of the current activations based on previous activations
        Recurrent SOMS perform better when their weight-based activation profile
        has high entropy, as small changes in context will then be able to have
        a larger effect.

        This is reflected in this function, which increases the importance of
        context by decreasing alpha if the entropy decreases. The function uses
        a very large momentum term of 0.9 to make sure the entropy does not rise
        or fall too sharply.

        :param prev_bmus: The previous BMUs.
        :param prev_update: The previous update, used as a momentum term.
        :return:
        """

        prev_bmus = np.array(list(prev_bmus.values()))
        prev_bmus = prev_bmus / np.sum(prev_bmus)

        new_entropy = -np.sum(prev_bmus * np.nan_to_num(np.log2(prev_bmus)))
        entropy_diff = (new_entropy - self.entropy)

        update = (entropy_diff * 0.1) + (prev_update * 0.9)

        self.entropy = new_entropy

        logger.info("Entropy: {0}".format(new_entropy))

        return update

    def _get_bmus(self, x, **kwargs):
        """
        Gets the best matching units, based on euclidean distance.
        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        # Differences is the components of the weights subtracted from the weight vector.
        differences_x = self._distance_difference(x, self.weights)
        # Idem for context.
        differences_y = self._distance_difference(kwargs['context'], self.context_weights)

        # Distances are squared euclidean norm of differences.
        # Since euclidean norm is sqrt(sum(square(x)))) we can leave out the sqrt
        # and avoid doing an extra square.
        distances_x = np.sum(np.square(differences_x), axis=1)
        distances_y = np.sum(np.square(differences_y), axis=1)

        # BMU is based on a weighted addition of current and previous activation.
        activations = ((1 - self.alpha) * distances_x) + (self.alpha * distances_y)

        return activations, differences_x, differences_y

    def _predict_base(self, X):
        """
        Predicts distances to some input data.

        :param X: The input data.
        :return: An array of arrays, representing the activation
        each node has to each input.
        """

        # Return the indices of the BMU which matches the input data most
        activations = []
        # context = np.zeros((self.data_dim,))
        prev_bmu = None

        for x in X:

            if prev_bmu is None:
                context = np.zeros((self.data_dim,))
            else:
                context = (1 - self.beta) * self.weights[prev_bmu] + self.beta * self.context_weights[prev_bmu]

            activation, _, _ = self._get_bmus(x, context=context)
            prev_bmu = np.argmin(activation)

            activations.append(activation)

        return activations
