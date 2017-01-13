import logging
import time
import numpy as np

from collections import defaultdict
from utils import progressbar, expo, linear, static
from functools import reduce


logger = logging.getLogger(__name__)


class Base(object):
    """
    Base is the base class from which both Soms and Neural gases are derived.
    It contains various functions which are agnostic to the inner workings of the
    specific models.

    This class can not be used on its own.
    """

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
        """

        :param map_dim: A tuple of map dimensions, e.g. (10, 10) instantiates a 10 by 10 map.
        :param dim: The data dimensionality.
        :param learning_rate: The learning rate, which is decreases according to some function
        :param lrfunc: The function to use in decreasing the learning rate. The functions are
        defined in utils. Default is exponential.
        :param nbfunc: The function to use in decreasing the neighborhood size. The functions
        are defined in utils. Default is exponential.
        :param calculate_distance_grid: The function to use in calculating the distance grid.
        This function should not be filled in by hand, and depends on the model (i.e. SOM versus NG)
        :param calculate_influence: Similar to above
        :param apply_influences: Similar to above
        :param sigma: The starting value for the neighborhood size, which is decreased over time.
        If sigma is None (default), sigma is calculated as ((max(map_dim) / 2) + 0.01), which is
        generally a good value.
        """

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

    def train(self, X, total_epochs=10, rough_epochs=1.0):
        """
        Fits the SOM to some data for a number of epochs.
        As the learning rate is decreased proportionally to the number
        of epochs, incrementally training a SOM is not feasible.

        :param X: the data on which to train.
        :param total_epochs: The number of epochs to simulate.
        :param rough_epochs: A fraction, describing over how many of the epochs
        the neighborhood should decrease. If the total number of epochs, for example
        is 1000, and rough_epochs = 0.5, the neighborhood size will be minimal after
        500 epochs.
        :return: None
        """

        # Calculate the number of updates required
        num_updates = total_epochs * rough_epochs

        # Decide on a value for lambda, depending on the function.
        if self.nbfunc != linear:
            self.lam = num_updates / np.log(self.sigma)
        else:
            self.lam = num_updates

        # The step size is the number of items between rough epochs.
        step_size = (X.shape[0] * rough_epochs) // num_updates

        # Precalculate the number of updates.
        update_counter = np.arange(step_size, (X.shape[0] * rough_epochs) + step_size, step_size)

        start = time.time()

        # Train
        self._train_loop(X, update_counter)
        self.trained = True

        logger.info("Total train time: {0}".format(time.time() - start))

    def _train_loop(self, X, update_counter):

        epoch = 0

        influences = self._param_update(0, len(update_counter))

        for idx, x in enumerate(progressbar(X)):

            self._example(x, influences)

            if idx in update_counter:
                epoch += 1

                influences = self._param_update(epoch, len(update_counter))

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

        return influences

    def _calc_influence(self, sigma):
        """


        :param sigma:
        :return:
        """

        raise NotImplementedError

    def _calculate_update(self, difference_vector, influence):
        """
        Updates the nodes, conditioned on the input vector,
        the influence, as calculated above, and the learning rate.

        Uses Oja's Rule: delta_W = alpha * (X - w)

        In this case (X - w) has been precomputed for speed, in the function
        _get_bmus.

        :param difference_vector: The difference between the input and some weights.
        :param influence: The influence the result has on each unit, depending on distance.
        Already includes the learning rate.
        """

        return difference_vector * influence

    def _get_bmus(self, x):
        """
        Gets the best matching units, based on euclidean distance.

        :param x: The input vector
        :return: The activations, which is a vector of map_dim, and
         the distances between the input and the weights, which can be
         reused in the update calculation.
        """

        differences = self._distance_difference(x, self.weights)

        # squared euclidean distance.
        activations = np.sqrt(np.sum(np.square(differences), axis=1))
        return activations, differences

    def _distance_difference(self, x, weights):
        """
        Calculates the difference between an input and all the weights.

        :param x: The input.
        :param weights: An array of weights.
        :return: A vector of differences.
        """
        return x - weights

    def _init_distance_grid(self):
        """


        :return:
        """

        raise NotImplementedError

    def _predict_base(self, X):
        """
        Predicts distances to some input data.

        :param X: The input data.
        :return: An array of arrays, representing the activation
        each node has to each input.
        """

        distances = []

        for x in X:
            distance, _ = self._get_bmus(x)
            distances.append(distance)

        return distances

    def quant_error(self, X):
        """
        Calculates the quantization error by taking the minimum euclidean
        distance between the units and some input.

        :param X: Input data.
        :return: A vector of numbers, representing the quantization error
        for each data point.
        """

        dist = self._predict_base(X)
        return np.min(dist, axis=1)

    def predict(self, X):
        """
        Predict the BMU for each input data.

        :param X: Input data.
        :return: The index of the bmu which best describes the input data.
        """

        dist = self._predict_base(X)
        return np.argmin(dist, axis=1)

    def receptive_field(self, X):
        """
        Calculate the receptive field of the SOM on some data.

        The receptive field is the common ending of all sequences which
        lead to the activation of a given BMU. If a SOM is well-tuned to
        specific sequences, it will have longer receptive fields, and therefore
        gives a better description of the dynamics of a given system.

        :param X: Input data.
        :return: The receptive field of each neuron.
        """

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
        """
        Calculate the inverted projection of a SOM by associating each
        unit with the input datum that gives the closest match.

        Works best for symbolic (instead of continuous) input data.

        :param X: Input data
        :param identities: The identities for each input datum, must be same length as X
        :return: A numpy array with identities, the shape of the map.
        """

        assert len(X) == len(identities)

        # Remove all duplicates from X
        X_unique, names = zip(*set([tuple((tuple(s), n)) for s, n in zip(X, identities)]))
        node_match = []

        for node in self.weights:

            differences = self._distance_difference(node, X_unique)
            distances = np.sqrt(np.sum(np.square(differences), axis=1))
            node_match.append(names[np.argmin(distances)])

        return np.array(node_match).reshape(self.map_dimensions)