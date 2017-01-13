import numpy as np
import time
import logging

from somber.som import Som
from somber.utils import expo, progressbar


logger = logging.getLogger(__name__)


class Merging(Som):

    def __init__(self, map_dim, dim, learning_rate, alpha, beta, sigma=None, lrfunc=expo, nbfunc=expo):

        super().__init__(map_dim, dim, learning_rate, lrfunc, nbfunc, sigma=sigma)

        self.alpha = alpha
        self.beta = beta
        self.context_weights = np.ones(self.weights.shape)
        self.entropy = 0

    def _train_loop(self, X, update_counter, context_mask):

        self.context_weights *= X.mean(axis=0)

        epoch = 0
        bmu = None
        influences = self._param_update(0, len(update_counter))

        bmu_entropy = np.zeros((self.map_dim,))
        update = 0

        for idx, x in enumerate(progressbar(X)):

            bmu = self._example(x, influences, prev_bmu=bmu)
            bmu_entropy[bmu] += 1

            if idx in update_counter:

                update = self._entropy(prev_bmus=bmu_entropy, prev_update=update)
                self.alpha += update
                self.alpha = np.clip(self.alpha, 0.0, 1.0)

                epoch += 1
                influences = self._param_update(epoch, len(update_counter))

            if idx in context_mask:

                bmu = None

    def _example(self, x, influences, **kwargs):
        """
        A single epoch.
        :param X: a numpy array of data
        :param map_radius: The radius at the current epoch, given the learning rate and map size
        :param learning_rates: The learning rate.
        :param batch_size: The batch size
        :return: The best matching unit
        """

        # context = kwargs['context']
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

        prev_bmus /= prev_bmus.sum()

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
