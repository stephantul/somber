import numpy as np
import time
import logging
import cProfile

from progressbar import progressbar
from som import Som, expo, static


logger = logging.getLogger(__name__)


class MSom(Som):

    def __init__(self, width, height, dim, learning_rate, alpha, beta, lrfunc=expo, nbfunc=expo):

        self.alpha = alpha
        self.beta = beta
        super().__init__(width, height, dim, learning_rate, lrfunc, nbfunc)

        self.context_weights = np.zeros_like(self.weights)
        self.entropy = 0

    def train(self, X, num_epochs=10, batch_size=100):
        """
        Fits the SOM to some data for a number of epochs.
        As the learning rate is decreased proportionally to the number
        of epochs, incrementally training a SOM is not feasible.

        :param X: the data on which to train.
        :param num_epochs: The number of epochs to simulate
        :return: None
        """

        # Scaler ensures that the neighborhood radius is 0 at the end of training
        # given a square map.
        self.lam = num_epochs / np.log(self.sigma)

        # Local copy of learning rate.
        learning_rate = self.learning_rates

        bmus = []

        real_start = time.time()

        print(X.shape)

        context = np.zeros((self.data_dim,))

        for epoch in range(num_epochs):

            print("\nEPOCH: {0}/{1}".format(epoch+1, num_epochs))
            start = time.time()

            map_radius = self.nbfunc(self.sigma, epoch, self.lam)
            context, distance = self.epoch_step(X, map_radius, learning_rate, batch_size=batch_size, context=context)

            learning_rate = self.lrfunc(self.learning_rates, epoch, num_epochs)

            print("\nEPOCH TOOK {0:.2f} SECONDS.".format(time.time() - start))
            print("TOTAL: {0:.2f} SECONDS.".format(time.time() - real_start))

            if epoch > (num_epochs / 2) and epoch % 2 == 0:

                self._entropy(distance / distance.sum())
                print(self.entropy)
                print(self.alpha)

        self.trained = True

        return bmus

    def epoch_step(self, X, map_radius, learning_rate, batch_size, **kwargs):
        """
        A single epoch.
        :param X: a numpy array of data
        :param map_radius: The radius at the current epoch, given the learning rate and map size
        :param learning_rate: The learning rate.
        :param batch_size: The batch size
        :return: The best matching unit
        """

        # Calc once per epoch
        lr = learning_rate[0]

        context = kwargs['context']

        influences = self._distance_grid(map_radius) * lr
        influences = np.asarray([influences] * self.data_dim).transpose((1, 2, 0))

        prev_bmu = 0

        for item in progressbar(X, idx_interval=1):

            distance, x_activations, y_activations = self._get_bmus(item, c=context)

            bmu = np.argmin(distance)

            context = (1 - self.beta) * self.weights[prev_bmu] + self.beta * self.context_weights[prev_bmu]

            influences_local = influences[bmu]

            update_x = self._calculate_update(x_activations, influences_local)
            update_y = self._calculate_update(y_activations, influences_local)

            self.weights += update_x
            self.context_weights += update_y

            prev_bmu = bmu

        return context, distance

    def _entropy(self, activation):

        new_entropy = -np.sum(activation * np.nan_to_num(np.log(activation+0.0001)))

        if new_entropy < self.entropy:
            self.alpha += 0.01
        else:
            self.alpha -= 0.01

        self.alpha = max(0.0, min(1.0, self.alpha))
        self.entropy = new_entropy

    def _get_bmus(self, x, **kwargs):
        """
        Gets the best matching units, based on euclidean distance.
        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        # Differences is the components of the weights subtracted from the weight vector.
        differences_weight = x - self.weights
        # Idem for the context.
        differences_context = kwargs['c'] - self.context_weights

        # Distances are squared euclidean norm of differences.
        # Since euclidean norm is sqrt(sum(square(x)))) we can leave out the sqrt
        # and avoid doing an extra square.
        activation_weight = np.sum(np.square(differences_weight), axis=1)
        activation_context = np.sum(np.square(differences_context), axis=1)

        # BMU is based on a weigted subtraction of current and the context
        activation = ((1 - self.alpha) * activation_weight) + (self.alpha * activation_context)

        return activation, differences_weight, differences_context

    def predict(self, X):
        """
        Predicts node identity for input data.
        Similar to a clustering procedure.
        :param x: The input data.
        :return: A list of indices
        """

        # Start with a clean buffer.
        context = np.zeros((self.data_dim,))

        all_bmus = []

        for item in X:

            distance, _, _ = self._get_bmus(item, c=context)
            bmus = np.argmin(distance)

            context = (1 - self.beta) * self.weights[bmus] + self.beta * self.context_weights[bmus]

            all_bmus.append(bmus)

        return np.array(all_bmus).T

    def _map(self, distance):
        """


        :param distance:
        :return:
        """

        return distance.reshape(self.width, self.height)

    def predict_distribution(self, X):
        """
        Predicts node identity for input data.
        Similar to a clustering procedure.
        :param x: The input data.
        :return: A list of indices
        """

        # Start with a clean buffer.
        context = np.zeros((X.shape[0], self.data_dim))

        all_bmus = []

        for idx in range(X.shape[1]):

            column = X[:, idx, :]
            distance, _, _ = self._get_bmus(column, c=context)

            bmus = np.argmin(distance)

            context = self.beta * self.weights[bmus] + (1 - self.beta) * self.context_weights[bmus]

            all_bmus.append(distance)

        return np.array(all_bmus).T


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    colors = np.array(
         [[1., 0., 1.],
          [0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])

    data = np.tile(colors, (100, 1, 1))

    colorpicker = np.arange(len(colors))

    d1 = colors[np.random.choice(colorpicker, size=15)]
    d2 = colors[np.random.choice(colorpicker, size=15)]
    d3 = colors[np.random.choice(colorpicker, size=15)]
    data = np.array([d1, d2, d3] * 100)
    print(data.shape)

    s = MSom(5, 5, 3, [1.0], 0.0, 0.5)
    start = time.time()
    cProfile.run("s.train(data.reshape(data.shape[0] * data.shape[1], data.shape[2]), num_epochs=1000, batch_size=1)")

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))