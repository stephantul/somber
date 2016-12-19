import numpy as np

from som import Som
from utils import expo, progressbar


class RSom(Som):

    def __init__(self, width, height, dim, learning_rate, alpha, sigma=None, lrfunc=expo, nbfunc=expo):

        super().__init__(width, height, dim, learning_rate, lrfunc, nbfunc, sigma)
        self.alpha = alpha

    def train(self, X, num_effective_epochs=10):

        # Scaler ensures that the neighborhood radius is 0 at the end of training
        # given a square map.
        self.lam = num_effective_epochs / np.log(self.sigma)

        # Local copy of learning rate.
        influences, learning_rate = self._param_update(0, num_effective_epochs)

        epoch_counter = X.shape[0] / num_effective_epochs
        epoch = 0

        prev_activation = np.zeros((self.map_dim, self.data_dim))

        for idx, x in enumerate(progressbar(X)):

            prev_activation = self._example(x, influences, prev_activation=prev_activation)

            if idx % epoch_counter == 0:

                epoch += 1

                influences, learning_rate = self._param_update(epoch, num_effective_epochs)

        self.trained = True
        print("Number of training items: {0}".format(X.shape[0]))
        print("Number of items per epoch: {0}".format(epoch_counter))

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
        distance, prev_activation = self._get_bmus(x, prev_activation=prev_activation)

        bmu = np.argmin(distance)
        influences_local = influences[bmu]

        # Minibatch update of X and Y. Returns arrays of updates,
        # one for each example.
        self.weights += self._calculate_update(prev_activation, influences_local)

        return prev_activation

    def _get_bmus(self, x, **kwargs):
        """
        Gets the best matching units, based on euclidean distance.
        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        # Differences is the components of the weights subtracted from the weight vector.
        difference_x = self._pseudo_distance(x, self.weights)
        activation = ((1 - self.alpha) * kwargs['prev_activation']) + (self.alpha * difference_x)

        # Distances are squared euclidean norm of differences.
        # Since euclidean norm is sqrt(sum(square(x)))) we can leave out the sqrt
        # and avoid doing an extra square.
        # Axis 2 because we are doing minibatches.
        distances = np.sum(np.square(activation), axis=1)

        return distances, activation

    def _predict_base(self, X):
        """
        Predicts node identity for input data.
        Similar to a clustering procedure.

        :param x: The input data.
        :return: A list of indices
        """

        # Return the indices of the BMU which matches the input data most
        distances = []

        prev_activation = np.zeros((self.map_dim, self.data_dim))

        for x in X:
            distance, prev_activation = self._get_bmus(x, prev_activation=prev_activation)
            distances.append(distance)

        return distances

