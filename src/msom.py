import numpy as np

from som import Som
from utils import expo, progressbar


class MSom(Som):

    def __init__(self, width, height, dim, learning_rate, alpha, beta, lrfunc=expo, nbfunc=expo):

        self.alpha = alpha
        self.beta = beta
        super().__init__(width, height, dim, learning_rate, lrfunc, nbfunc)

        self.context_weights = np.zeros_like(self.weights)
        self.entropy = 0

    def train(self, X, num_effective_epochs=10):

        # Scaler ensures that the neighborhood radius is 0 at the end of training
        # given a square map.
        self.lam = num_effective_epochs / np.log(self.sigma)

        # Local copy of learning rate.
        influences, learning_rates = self._param_update(0, num_effective_epochs)

        epoch_counter = X.shape[0] // num_effective_epochs
        epoch = 0

        context = np.zeros((self.data_dim,))
        bmu = 0

        update = 0

        for idx, x in enumerate(progressbar(X)):

            context, bmu = self._example(x, influences, prev_bmu=bmu, context=context)

            if idx % epoch_counter == 0:

                epoch += 1

                influences, learning_rates = self._param_update(epoch, num_effective_epochs)
                update = self._entropy(context, update)

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

        context = kwargs['context']
        prev_bmu = kwargs['prev_bmu']

        # Get the indices of the Best Matching Units, given the data.
        distance, x_activations, y_activations = self._get_bmus(x, context=context)

        bmu = np.argmin(distance)

        context = (1 - self.beta) * self.weights[prev_bmu] + self.beta * self.context_weights[prev_bmu]
        influences_local = influences[bmu]

        # Minibatch update of X and Y. Returns arrays of updates,
        # one for each example.
        self.weights += self._calculate_update(x_activations, influences_local)
        self.context_weights += self._calculate_update(y_activations, influences_local)

        return context, bmu

    def _entropy(self, context, prev_update):

        new_entropy = -np.sum(context * np.nan_to_num(np.log2(context)))

        print(new_entropy, self.entropy)

        update = (new_entropy - self.entropy) * 0.1

        print(update, prev_update)

        self.alpha -= update + (prev_update * 0.5)

        print(self.alpha)
        self.alpha = max(0.0, min(1.0, self.alpha))
        self.entropy = new_entropy

        return update

    def _get_bmus(self, x, **kwargs):
        """
        Gets the best matching units, based on euclidean distance.
        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        # Differences is the components of the weights subtracted from the weight vector.
        differences_x = self._pseudo_distance(x, self.weights)
        # Idem for previous activation.
        differences_y = self._pseudo_distance(kwargs['context'], self.context_weights)

        # Distances are squared euclidean norm of differences.
        # Since euclidean norm is sqrt(sum(square(x)))) we can leave out the sqrt
        # and avoid doing an extra square.
        # Axis 2 because we are doing minibatches.
        distances_x = np.sum(np.square(differences_x), axis=1)
        distances_y = np.sum(np.square(differences_y), axis=1)

        # BMU is based on a weigted subtraction of current and previous activation.
        distances = ((1 - self.alpha) * distances_x) + (self.alpha * distances_y)

        return distances, differences_x, differences_y

    def _predict_base(self, X):
        """
        Predicts node identity for input data.
        Similar to a clustering procedure.

        :param x: The input data.
        :return: A list of indices
        """

        # Return the indices of the BMU which matches the input data most
        distances = []
        context = np.zeros((self.data_dim,))
        prev_bmu = 0

        for x in X:
            distance, _, _ = self._get_bmus(x, context=context)
            context = (1 - self.beta) * self.weights[prev_bmu] + self.beta * self.context_weights[prev_bmu]
            bmu = np.argmin(distance)
            prev_bmu = bmu

            distances.append(distance)

        return distances
