import logging
import time

import json
import numpy as np

from ..utils import progressbar, linear, expo, np_min, resize
from functools import reduce
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class Som(object):
    """
    This is the batched version of the basic SOM.
    """

    def __init__(self,
                 map_dim,
                 data_dim,
                 learning_rate,
                 lrfunc=expo,
                 nbfunc=expo,
                 sigma=None,
                 min_max=np_min):
        """
        A batched Self-Organizing-Map.

        :param map_dim: A tuple describing the MAP size.
        :param data_dim: The dimensionality of the input matrix.
        :param learning_rate: The learning rate.
        :param sigma: The neighborhood factor.
        :param lrfunc: The function used to decrease the learning rate.
        :param nbfunc: The function used to decrease the neighborhood
        :param min_max: The function used to determine the winner.
        """
        if sigma is not None:
            self.sigma = sigma
        else:
            # Add small constant to sigma to prevent
            # divide by zero for maps of size 2.
            self.sigma = (max(map_dim) / 2.0) + 0.01

        self.learning_rate = learning_rate
        # A tuple of dimensions
        # Usually (width, height), but can accomodate one-dimensional maps.
        self.map_dimensions = map_dim

        # The dimensionality of the weight vector
        # Usually (width * height)
        self.weight_dim = int(reduce(np.multiply, map_dim, 1))

        # Weights are initialized to small random values.
        # Initializing to more appropriate values given the dataset
        # will probably give faster convergence.
        self.weights = np.zeros((self.weight_dim, data_dim))
        self.data_dim = data_dim

        # The function used to diminish the learning rate.
        self.lrfunc = lrfunc
        # The function used to diminish the neighborhood size.
        self.nbfunc = nbfunc

        # Initialize the distance grid: only needs to be done once.
        self.distance_grid = self._initialize_distance_grid()
        self.min_max = min_max
        self.trained = False

        self.progressbar_interval = 10
        self.progressbar_mult = 1

    def fit(self,
            X,
            num_epochs=10,
            init_pca=True,
            total_updates=50,
            stop_lr_updates=1.0,
            stop_nb_updates=1.0,
            batch_size=100,
            show_progressbar=False,
            seed=44):
        """
        Fit the SOM to some data.

        The updates correspond to the number of updates to the parameters
        (i.e. learning rate, neighborhood, not weights!)
        to perform during training.

        In general, 1000 updates will do for most learning problems.

        :param X: the data on which to train.
        :param num_epochs: the number of epochs for which to train.
        :param init_pca: Whether to initialize the weights to the
        first principal components of the input data.
        :param total_updates: The number of updates to the parameters to do
        during training.
        :param stop_lr_updates: A fraction, describing over which portion of
        the training data the learning rate should decrease. If the total
        number of updates, for example is 1000, and stop_updates = 0.5,
        1000 updates will have occurred after half of the examples.
        After this period, no updates of the parameters will occur.
        :param stop_nb_updates: A fraction, describing over which portion of
        the training data the neighborhood should decrease.
        :param batch_size: the batch size
        :param show_progressbar: whether to show the progress bar.
        :param seed: The random seed
        :return: None
        """

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError("Your data is not a 2D matrix. Actual size: {0}".format(X.shape))

        if X.shape[1] != self.data_dim:
            raise ValueError("Your data size != weight dim: {0}, expected {1}".format(X.shape[1], self.data_dim))

        np.random.seed(seed)

        if init_pca:
            min_ = np.min(X, axis=0)
            random = np.random.rand(self.weight_dim).reshape((self.weight_dim, 1))
            temp = np.outer(random, np.abs(np.max(X, axis=0) - min_))
            self.weights = np.asarray(min_ + temp, dtype=np.float32)

        # The train length
        train_length = (len(X) * num_epochs) // batch_size

        X = self._create_batches(X, batch_size)

        # The step size is the number of items between rough epochs.
        # We use len instead of shape because len also works with np.flatiter
        step_size_lr = max((train_length * stop_lr_updates) // total_updates, 1)
        step_size_nb = max((train_length * stop_nb_updates) // total_updates, 1)

        # Precalculate the number of updates.
        # Precalculate the number of updates.
        lr_update_counter = set(np.arange(step_size_lr,
                                          (train_length * stop_lr_updates) + step_size_lr,
                                          step_size_lr).tolist())

        nb_update_counter = set(np.arange(step_size_nb,
                                          (train_length * stop_nb_updates) + step_size_nb,
                                          step_size_nb).tolist())
        start = time.time()

        # Train
        nb_step = 0
        lr_step = 0
        idx = 0

        for epoch in range(num_epochs):

            logger.info("Epoch {0} of {1}".format(epoch, num_epochs))

            idx, nb_step, lr_step = self._epoch(X,
                                                nb_update_counter,
                                                lr_update_counter,
                                                idx, nb_step,
                                                lr_step,
                                                show_progressbar)

        self.trained = True

        logger.info("Total train time: {0}".format(time.time() - start))

    def _init_prev(self, X):
        """
        Placeholder

        :param X:
        :return:
        """
        return None

    def _epoch(self,
               X,
               nb_update_counter,
               lr_update_counter,
               idx,
               nb_step,
               lr_step,
               show_progressbar):
        """
        A single epoch.

        This function uses an index parameter which is passed around to see to
        how many training items the SOM has been exposed globally.

        nb and lr_update_counter hold the indices at which the neighborhood
        size and learning rates need to be updated.
        These are therefore also passed around. The nb_step and lr_step
        parameters indicate how many times the neighborhood
        and learning rate parameters have been updated already.

        :param X: The training data.
        :param nb_update_counter: The indices at which to
        update the neighborhood.
        :param lr_update_counter: The indices at which to
        update the learning rate.
        :param idx: The current index.
        :param nb_step: The current neighborhood step.
        :param lr_step: The current learning rate step.
        :param show_progressbar: Whether to show a progress bar or not.
        :return: The index, neighborhood step and learning rate step
        """
        # Initialize the previous activation
        prev_activation = self._init_prev(X)

        # Calculate the influences for update 0.
        map_radius = self.nbfunc(self.sigma,
                                 nb_step,
                                 len(nb_update_counter))

        learning_rate = self.lrfunc(self.learning_rate,
                                    lr_step,
                                    len(lr_update_counter))

        influences = self._calculate_influence(map_radius) * learning_rate

        # Iterate over the training data
        for x in progressbar(X,
                             use=show_progressbar,
                             mult=self.progressbar_mult,
                             idx_interval=self.progressbar_interval):

            prev_activation = self._propagate(x,
                                              influences,
                                              prev_activation=prev_activation)

            if idx in nb_update_counter:
                nb_step += 1

                map_radius = self.nbfunc(self.sigma,
                                         nb_step,
                                         len(nb_update_counter))
                logger.info("Updated map radius: {0}".format(map_radius))

                # The map radius has been updated, so the influence
                # needs to be recalculated
                influences = self._calculate_influence(map_radius)
                influences *= learning_rate

            if idx in lr_update_counter:
                lr_step += 1

                # Reset the influences back to 1
                influences /= learning_rate
                learning_rate = self.lrfunc(self.learning_rate,
                                            lr_step,
                                            len(lr_update_counter))
                logger.info("Updated learning rate: {0}".format(learning_rate))
                # Recalculate the influences
                influences *= learning_rate

            idx += 1

        return idx, nb_step, lr_step

    def _create_batches(self, X, batch_size):
        """
        Create batches out of a sequence of data.

        Assumes ndim(X) == 2.

        This function will append zeros to the end of your data to ensure that
        all batches are even-sized.

        :param X: A numpy array, representing your input data.
        Must have 2 dimensions.
        :param batch_size: The desired batch size.
        :return: A batched version of your data.
        """
        self.progressbar_interval = 1
        self.progressbar_mult = batch_size

        max_x = int(np.ceil(X.shape[0] / batch_size))
        X = resize(X, (max_x, batch_size, self.data_dim))

        return X

    def _propagate(self, x, influences, **kwargs):
        """
        Propagate a single example through the network, and update the
        weights based on the response and map parameters.

        :param X: a numpy array of data
        :param influences: The influence at the current epoch,
        given the learning rate and map size
        :return: A vector describing activation values for each unit.
        """
        activation, difference_x = self.forward(x)
        self.backward(difference_x, influences, activation)

        return activation

    def forward(self, x, **kwargs):
        """
        Gets the best matching units, based on euclidean distance.

        :param x: The input vector
        :return: The activations, and
         the differences between the input and the weights, which can be
         reused in the update calculation.
        """
        return self.distance_function(x, self.weights)

    def _calculate_update(self, x, influence):
        """
        Multiply the difference vector with the influence vector.

        The influence vector is chosen based on the BMU, so that the update
        done to the every weight node is proportional to its distance from the
        BMU.

        Implicitly uses Oja's Rule: delta_W = alpha * (X - w)

        In this case (X - w) has been precomputed for speed, in the
        forward step.

        :param x: The input vector
        :param influence: The influence the result has on each unit,
        depending on distance. Already includes the learning rate.
        """
        return np.multiply(x, influence)

    def backward(self, x, influences, activation, **kwargs):
        """
        Backward pass through the network, including update.

        :param x: The input data
        :param influences: The influences at the current time-step
        :param activation: The activation at the output layer
        :param kwargs:
        :return: None
        """

        influence = self._apply_influences(activation, influences)
        self.weights += self._calculate_update(x, influence).mean(0)

    def distance_function(self, x, weights):
        """
        batched version of the euclidean distance.

        :param x: The input
        :param weights: The weights
        :return: A tuple of matrices,
        The first matrix is a (batch_size * neurons) matrix of
        activation values, containing the response of each neuron
        to each input
        The second matrix is a (batch_size * neuron) matrix containing
        the difference between euch neuron and each input.
        """
        diff = self._distance_difference(x, weights)
        activations = np.linalg.norm(diff, axis=-1)

        return activations, diff

    def _distance_difference(self, x, weights):
        """
        Calculate the difference between an input and all the weights.

        :param x: The input.
        :param weights: An array of weights.
        :return: A vector of differences.
        """
        return x[:, None, :] - weights[None, :, :]

    def _apply_influences(self, activations, influences):
        """
        Calculate the BMU using min_max, and get the appropriate influences.

        Then gets the appropriate influence from the neighborhood,
        given the BMU

        :param activations: A Numpy array of distances.
        :param influences: A (map_dim, map_dim, data_dim) array
        describing the influence each node has on each other node.
        :return: The influence given the bmu, and the index of the bmu itself.
        """
        bmu = self.min_max(activations, 1)[1]
        return influences[bmu]

    def _calculate_influence(self, sigma):
        """
        Pre-calculate the influence for a given value of sigma.

        The neighborhood has size map_dim * map_dim, so for a 30 * 30 map,
        the neighborhood will be size (900, 900). It is then duplicated
        influence_size times, and reshaped into an
        (map_dim, map_dim, influence_size) array.
        This is done to facilitate fast calculation in subsequent steps.

        :param sigma: The neighborhood value.
        :return: The neighborhood, reshaped into an array
        """
        neighborhood = np.exp(-self.distance_grid / (2.0 * sigma ** 2))
        return neighborhood.reshape(self.weight_dim, self.weight_dim)[:, :, None]

    def _initialize_distance_grid(self):
        """
        Initialize the distance grid by calls to _grid_dist.

        :return:
        """
        p = [self._grid_distance(i) for i in range(self.weight_dim)]
        return np.array(p, dtype=np.float32)

    def _grid_distance(self, index):
        """
        Calculate the distance grid for a single index position.

        This is pre-calculated for fast neighborhood calculations
        later on (see _calc_influence).

        :param index: The index for which to calculate the distances.
        :return: A flattened version of the distance array.
        """
        width, height = self.map_dimensions

        row = index // width
        column = index % width

        # Fast way to construct distance matrix
        f = np.arange(0, self.weight_dim).reshape(self.map_dimensions)
        x = np.abs(f % width - row)
        y = np.abs(f % height - column).transpose(1, 0)

        distance = x + y

        return distance.ravel()

    def _predict_base(self, X, batch_size=100):
        """
        Predict distances to some input data.

        This function should not be directly used.

        :param X: The input data.
        :return: A matrix, representing the activation
        each node has to each input.
        """

        batched = self._create_batches(X, batch_size)

        activations = []

        for x in batched:
            activation = self.forward(x)[0]
            activations.extend(activation)

        activations = np.asarray(activations, dtype=np.float32)
        activations = activations[:X.shape[0]]
        return activations.reshape(X.shape[0], self.weight_dim)

    def predict(self, X, batch_size=100):
        """
        Predict the BMU for each input data.

        :param X: Input data.
        :param batch_size: The batch size to use in prediction.
        :return: The index of the bmu which best describes the input data.
        """
        dist = self._predict_base(X, batch_size)
        return self.min_max(dist, 1)[1]

    def quant_error(self, X):
        """
        Calculate the quantization error.

        Find the the minimum euclidean distance between the units and
        some input.

        :param X: Input data.
        :return: A vector of numbers, representing the quantization error
        for each data point.
        """
        dist = self._predict_base(X)
        return self.min_max(dist, 1)[0]

    def receptive_field(self, X, identities, max_len=10, threshold=0.9, batch_size=1):
        """
        Calculate the receptive field of the SOM on some data.

        The receptive field is the common ending of all sequences which
        lead to the activation of a given BMU. If a SOM is well-tuned to
        specific sequences, it will have longer receptive fields, and therefore
        gives a better description of the dynamics of a given system.

        :param X: Input data.
        :param identities: The letters associated with each input datum.
        :param max_len: The maximum length sequence we expect.
        Increasing the window size leads to accurate results,
        but costs more memory.
        :param threshold: The threshold at which we consider a receptive field as being valid.
        If at least this proportion of the sequences of a neuron have the same suffix, that suffix
        is counted as acquired.
        :param batch_size: The batch size to use in prediction
        :return: The receptive field of each neuron.
        """
        assert len(X) == len(identities)

        receptive_fields = defaultdict(list)
        predictions = self.predict(X, batch_size)

        for idx, p in enumerate(predictions.tolist()):
            receptive_fields[p].append(identities[idx+1 - max_len:idx+1])

        sequence = defaultdict(list)

        for k, v in receptive_fields.items():

            v = [x for x in v if np.any(x)]

            total = len(v)
            v = ["".join([str(x_) for x_ in np.squeeze(x)]) for x in v]

            for row in reversed(list(zip(*v))):

                r = Counter(row)

                for _, count in r.items():
                    if count / total > threshold:
                        sequence[k].append(row[0])
                    else:
                        break

        return {k: v[::-1] for k, v in sequence.items()}

    def invert_projection(self, X, identities):
        """
        Calculate the inverted projection.

        The inverted projectio of a SOM is created by association each weight
        with the input datum which matches it the most.
        Works best for symbolic (instead of continuous) input data.

        :param X: Input data
        :param identities: The identities for each
        input datum, must be same length as X
        :return: A numpy array with identities, the shape of the map.
        """
        if len(X) != len(identities):
            raise ValueError("X and identities are not the same length: {0} and {1}".format(len(X), len(identities)))

        # Remove all duplicates from X
        X_unique, names = zip(*set([tuple((tuple(s), n)) for s, n in zip(X, identities)]))
        node_match = []

        X_unique = np.array(X_unique, dtype=np.float32)

        for node in self.weights:

            differences = node - X_unique
            distances = np.sum(np.square(differences), 1)
            node_match.append(names[np.argmin(distances)])

        return np.array(node_match, dtype=np.float32)

    def map_weights(self):
        """
        Retrieve the grid as a list of lists of weights.

        :return: A three-dimensional Numpy array of values.
        """
        width, height = self.map_dimensions
        # Reshape to appropriate dimensions
        view = self.weights.reshape((width, height, self.data_dim))
        return view.T

    @classmethod
    def load(cls, path):
        """
        Loads a SOM from a JSON file.

        :param path: The path to the JSON file.
        :return: A SOM.
        """

        data = json.load(open(path))

        weights = data['weights']
        weights = np.asarray(weights, dtype=np.float32)
        datadim = weights.shape[1]

        dimensions = data['dimensions']
        lrfunc = expo if data['lrfunc'] == 'expo' else linear
        nbfunc = expo if data['nbfunc'] == 'expo' else linear
        lr = data['lr']
        sigma = data['sigma']

        s = cls(dimensions, datadim, lr, lrfunc=lrfunc, nbfunc=nbfunc, sigma=sigma)
        s.weights = weights
        s.trained = True

        return s

    def save(self, path):
        """
        Saves a SOM to a JSON file.

        :param path: The path to the JSON file that will be created
        :return: None
        """

        to_save = {}
        to_save['weights'] = [[float(w) for w in x] for x in self.weights]
        to_save['dimensions'] = self.map_dimensions
        to_save['lrfunc'] = 'expo' if self.lrfunc == expo else 'linear'
        to_save['nbfunc'] = 'expo' if self.nbfunc == expo else 'linear'
        to_save['lr'] = self.learning_rate
        to_save['sigma'] = self.sigma

        json.dump(to_save, open(path, 'w'))
