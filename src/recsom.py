import numpy as np

from som import Som
from utils import expo, progressbar


class RecSom(Som):

    def __init__(self, width, height, dim, learning_rate, alpha, beta, sigma=None, lrfunc=expo, nbfunc=expo):

        super().__init__(width, height, dim, learning_rate, lrfunc, nbfunc, sigma)
        self.context_weights = np.random.uniform(0.0, 1.0, (self.map_dim, self.map_dim))
        self.alpha = alpha
        self.beta = beta

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
        activation, diff_x, diff_y = self._get_bmus(x, prev_activation=prev_activation)

        bmu = np.argmin(activation)
        influences_local = influences[bmu]

        # Minibatch update of X and Y. Returns arrays of updates,
        # one for each example.
        self.weights += self._calculate_update(diff_x, influences_local)
        self.context_weights += self._calculate_update(diff_y, influences_local)

        return activation

    def _get_bmus(self, x, **kwargs):
        """
        Gets the best matching units, based on euclidean distance.
        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        prev_activation = kwargs['prev_activation']

        # Differences is the components of the weights subtracted from the weight vector.
        difference_x = self._pseudo_distance(x, self.weights)
        difference_y = self._pseudo_distance(prev_activation, self.context_weights)

        # Distances are squared euclidean norm of differences.
        # Since euclidean norm is sqrt(sum(square(x)))) we can leave out the sqrt
        # and avoid doing an extra square.
        # Axis 2 because we are doing minibatches.
        distance_x = np.sum(np.square(difference_x), axis=1)
        distance_y = np.sum(np.square(difference_y), axis=1)

        distances = np.exp(-(self.alpha * distance_x) - self.beta * distance_y)

        return distances, difference_x, difference_y

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
            prev_activation, _, _ = self._get_bmus(x, prev_activation=prev_activation)
            distances.append(prev_activation)

        return distances


