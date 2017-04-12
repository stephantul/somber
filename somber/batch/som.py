import logging
import time
import numpy as np

from somber.utils import progressbar, expo, linear, static
from somber.som import Som as Base_Som


logger = logging.getLogger(__name__)


class Som(Base_Som):
    """
    This is the basic SOM class.
    """

    def __init__(self,
                 map_dim,
                 dim,
                 learning_rate,
                 lrfunc=expo,
                 nbfunc=expo,
                 sigma=None,
                 min_max=np.argmin):
        """

        :param map_dim: A tuple of map dimensions, e.g. (10, 10) instantiates a 10 by 10 map.
        :param dim: The data dimensionality.
        :param learning_rate: The learning rate, which is decreases according to some function
        :param lrfunc: The function to use in decreasing the learning rate. The functions are
        defined in utils. Default is exponential.
        :param nbfunc: The function to use in decreasing the neighborhood size. The functions
        are defined in utils. Default is exponential.
        :param sigma: The starting value for the neighborhood size, which is decreased over time.
        If sigma is None (default), sigma is calculated as ((max(map_dim) / 2) + 0.01), which is
        generally a good value.
        """

        super().__init__(map_dim=map_dim,
                         dim=dim,
                         learning_rate=learning_rate,
                         lrfunc=lrfunc,
                         nbfunc=nbfunc,
                         sigma=sigma,
                         min_max=min_max)

    def train(self, X, num_epochs=10, total_updates=10, stop_lr_updates=1.0, stop_nb_updates=1.0, context_mask=(), batch_size=100, show_progressbar=False):
        """
        Fits the SOM to some data.
        The updates correspond to the number of updates to the parameters
        (i.e. learning rate, neighborhood, not weights!) to perform during training.

        In general, 1000 updates will do for most learning problems.

        :param X: the data on which to train.
        :param total_updates: The number of updates to the parameters to do during training.
        :param stop_lr_updates: A fraction, describing over which portion of the training data
        the neighborhood and learning rate should decrease. If the total number of updates, for example
        is 1000, and stop_updates = 0.5, 1000 updates will have occurred after half of the examples.
        After this period, no updates of the parameters will occur.
        :param context_mask: a binary mask used to indicate whether the context should be set to 0
        at that specified point in time. Used to make items conditionally independent on previous items.
        Examples: Spaces in character-based models of language. Periods and question marks in models of sentences.
        :return: None
        """

        if not self.trained:
            min_ = np.min(X, axis=0)
            random = np.random.rand(self.map_dim).reshape((self.map_dim, 1))
            temp = np.outer(random, np.abs(np.max(X, axis=0) - min_))
            self.weights = min_ + temp

        # The train length
        train_length = (len(X) * num_epochs) // batch_size

        if not np.any(context_mask):
            context_mask = np.ones((len(X), 1))

        X = self._create_batches(X, batch_size)
        context_mask = self._create_batches(context_mask, batch_size)

        # The step size is the number of items between rough epochs.
        # We use len instead of shape because len also works with np.flatiter
        step_size_lr = max((train_length * stop_lr_updates) // total_updates, 1)
        step_size_nb = max((train_length * stop_nb_updates) // total_updates, 1)

        # Precalculate the number of updates.
        lr_update_counter = np.arange(step_size_lr, (train_length * stop_lr_updates) + step_size_lr, step_size_lr)
        nb_update_counter = np.arange(step_size_nb, (train_length * stop_nb_updates) + step_size_nb, step_size_nb)

        start = time.time()

        # Train
        self._train_loop(X, num_epochs, lr_update_counter, nb_update_counter, context_mask, show_progressbar)
        self.trained = True

        logger.info("Total train time: {0}".format(time.time() - start))

    def _create_batches(self, X, batch_size):
        """
        Creates batches out of a sequential piece of data.
        Assumes ndim(X) == 2.

        This function will append zeros to the end of your data to make all batches even-sized.

        :param X: A numpy array, representing your input data. Must have 2 dimensions.
        :param batch_size: The desired batch size.
        :return: A batched version of your data.
        """

        return np.resize(X, (int(np.ceil(X.shape[0] / batch_size)), batch_size, X.shape[1]))

    def _example(self, x, influences, **kwargs):
        """
        A single example.

        :param x: a single example
        :param influences: an array with influence values.
        :return: The best matching unit
        """

        # activation, difference_x = self._get_bmus(x)
        activation, difference_x = self._get_bmus(x)

        influences, bmu = self._apply_influences(activation, influences)
        self.weights += self._calculate_update(difference_x, influences).mean(axis=0)

        return bmu

    def _get_bmus(self, x):
        """
        Gets the best matching units, based on euclidean distance.

        :param x: The input vector
        :return: The activations, which is a vector of map_dim, and
         the distances between the input and the weights, which can be
         reused in the update calculation.
        """

        diff = self._distance_difference(x, self.weights)
        euclidean = self._euclidean(x, self.weights)

        return euclidean, diff

    def _euclidean(self, x, weights):

        m_norm = np.square(x).sum(axis=1)
        w_norm = np.square(weights).sum(axis=1)[:, np.newaxis]
        dotted = np.dot(np.multiply(x, 2), weights.T)

        res = np.outer(m_norm, np.ones((1, w_norm.shape[0])))
        res += np.outer(np.ones((m_norm.shape[0], 1)), w_norm.T)
        res -= dotted

        return res

    def _distance_difference(self, x, weights):
        """
        Calculates the difference between an input and all the weights.

        :param x: The input.
        :param weights: An array of weights.
        :return: A vector of differences.
        """

        return np.array([v - weights for v in x])

    def _predict_base(self, X):
        """
        Predicts distances to some input data.

        :param X: The input data.
        :return: An array of arrays, representing the activation
        each node has to each input.
        """

        X = self._create_batches(X, 1)

        distances = []

        for x in X:
            distance, _ = self._get_bmus(x)
            distances.extend(distance)

        return np.array(distances)

    def _apply_influences(self, activations, influences):
        """
        First calculates the BMU.
        Then gets the appropriate influence from the neighborhood, given the BMU

        :param activations: A Numpy array of distances.
        :param influences: A (map_dim, map_dim, data_dim) array describing the influence
        each node has on each other node.
        :return: The influence given the bmu, and the index of the bmu itself.
        """

        bmu = self.min_max(activations, axis=1)
        return influences[bmu], bmu
