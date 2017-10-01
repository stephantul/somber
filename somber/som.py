import logging
import time
import types
import json
import cupy as cp
import numpy as np

from tqdm import tqdm
from .components.utilities import linear, expo, resize, Scaler, shuffle
from .components.initializers import range_initialization
from collections import Counter, defaultdict


logger = logging.getLogger(__name__)


class Som(object):
    """This is a batched version of the basic SOM."""

    # Static property names
    param_names = {'neighborhood',
                   'learning_rate',
                   'map_dimensions',
                   'weights',
                   'data_dimensionality',
                   'lrfunc',
                   'nbfunc',
                   'valfunc',
                   'argfunc'}

    def __init__(self,
                 map_dimensions,
                 data_dimensionality,
                 learning_rate,
                 lrfunc=expo,
                 nbfunc=expo,
                 neighborhood=None,
                 argfunc="argmin",
                 valfunc="min",
                 initializer=range_initialization):
        """
        A batched Self-Organizing-Map.

        :param map_dimensions: A tuple describing the MAP size.
        :param data_dimensionality: The dimensionality of the input matrix.
        :param learning_rate: The learning rate.
        :param sigma: The neighborhood factor.
        :param lrfunc: The function used to decrease the learning rate.
        :param nbfunc: The function used to decrease the neighborhood
        :param min_max: The function used to determine the winner.
        """
        if neighborhood is not None:
            self.neighborhood = neighborhood
        else:
            # Add small constant to sigma to prevent
            # divide by zero for maps with the same max_dim as the number
            # of dimensions.
            self.neighborhood = max(map_dimensions) / 2
            self.neighborhood += 0.0001

        self.learning_rate = learning_rate
        # A tuple of dimensions
        # Usually (width, height), but can accomodate N-dimensional maps.
        self.map_dimensions = map_dimensions

        self.weight_dim = np.int(np.prod(map_dimensions))
        self.weights = np.zeros((self.weight_dim, data_dimensionality))
        self.data_dimensionality = data_dimensionality

        # The function used to diminish the learning rate.
        self.lrfunc = lrfunc
        # The function used to diminish the neighborhood size.
        self.nbfunc = nbfunc

        # Initialize the distance grid: only needs to be done once.
        self.distance_grid = self._initialize_distance_grid()
        self.argfunc = argfunc
        self.valfunc = valfunc
        self.trained = False
        self.scaler = Scaler()
        self.initializer = initializer

    def fit(self,
            X,
            num_epochs=10,
            updates_epoch=10,
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

        In general, 5 to 10 updates will do for most learning problems.
        Doing updates to the neighborhood gets more expensive as the
        size of the map increases. Hence, it is advisable to not make
        the number of updates too big for your learning problem.

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
        xp = cp.get_array_module(X)

        X = xp.asarray(X, dtype=xp.float32)
        self._check_input(X)

        X = self.scaler.fit_transform(X)
        xp.random.seed(seed)

        if self.initializer:
            self.weights = self.initializer(X, self.weights)
        self._ensure_params(X)

        start = time.time()

        total_nb_epochs = np.ceil(num_epochs * stop_nb_updates)
        total_lr_epochs = np.ceil(num_epochs * stop_lr_updates)

        for epoch in range(num_epochs):

            logger.info("Epoch {0} of {1}".format(epoch, num_epochs))
            X_ = self._create_batches(X, batch_size)

            self._epoch(X_,
                        epoch,
                        batch_size,
                        updates_epoch,
                        total_nb_epochs,
                        total_lr_epochs,
                        show_progressbar)

        self.trained = True
        self.weights = self.scaler.inverse_transform(self.weights)

        logger.info("Total train time: {0}".format(time.time() - start))

    def _epoch(self,
               X,
               epoch_idx,
               batch_size,
               update_steps,
               total_nb_epochs,
               total_lr_epochs,
               show_progressbar):
        """
        Run a single epoch.

        This function uses an index parameter which is passed around to see to
        how many training items the SOM has been exposed globally.

        nb and lr_update_counter hold the indices at which the neighborhood
        size and learning rates need to be updated.
        These are therefore also passed around. The nb_step and lr_step
        parameters indicate how many times the neighborhood
        and learning rate parameters have been updated already.

        :param X: The training data.
        :param nb_update_steps: The indices at which to
        update the neighborhood.
        :param lr_update_steps: The indices at which to
        update the learning rate.
        :param idx: The current index.
        :param nb_step: The current neighborhood step.
        :param lr_step: The current learning rate step.
        :param show_progressbar: Whether to show a progress bar or not.
        :return: The index, neighborhood step and learning rate step
        """
        num = 0

        total_steps_nb = update_steps * total_nb_epochs
        total_steps_lr = update_steps * total_lr_epochs
        current_step = update_steps * epoch_idx

        update_step = int(np.ceil(len(X) / update_steps))

        map_radius = self.nbfunc(self.neighborhood,
                                 current_step,
                                 total_steps_nb)
        influences = self._calculate_influence(map_radius)

        learning_rate = self.lrfunc(self.learning_rate,
                                    current_step,
                                    total_steps_lr)

        influences *= learning_rate

        # Initialize the previous activation
        prev_activation = self._init_prev(X)

        # Iterate over the training data
        for x in tqdm(X, disable=not show_progressbar):

            if num % update_step == 0 and current_step + num < total_steps_lr:

                prev_lr = learning_rate

                learning_rate = self.lrfunc(self.learning_rate,
                                            current_step + num,
                                            total_steps_lr)

                logger.info("Updated learning rate: {0}".format(learning_rate))
                # Recalculate the influences
                influences *= (learning_rate / prev_lr)

            if num % update_step == 0 and current_step + num < total_steps_nb:

                # Exponential decay is based on the number of steps
                # and max neighborhood size.
                # factor = len(nb_update_steps) / np.log(self.neighborhood)
                map_radius = self.nbfunc(self.neighborhood,
                                         current_step + num,
                                         total_steps_nb)
                logger.info("Updated map radius: {0}".format(map_radius))

                # The map radius has been updated, so the influence
                # needs to be recalculated
                influences = self._calculate_influence(map_radius)
                influences *= learning_rate

            prev_activation = self._propagate(x,
                                              influences,
                                              prev_activation=prev_activation)

            num += 1

    def _ensure_params(self, X):
        """
        Ensure the parameters are of the correct type.

        :param X: The input data
        :return: None
        """
        xp = cp.get_array_module(X)
        self.weights = xp.asarray(self.weights, xp.float32)
        self.distance_grid = xp.asarray(self.distance_grid, xp.int32)

    def _init_prev(self, x):
        """
        Placeholder.

        :param x:
        :return:
        """
        return None

    def _create_batches(self, X, batch_size, shuffle_data=True):
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
        xp = cp.get_array_module(X)

        if shuffle_data:
            X = shuffle(X)

        if batch_size > X.shape[0]:
            batch_size = X.shape[0]

        max_x = int(xp.ceil(X.shape[0] / batch_size))
        X = resize(X, (max_x, batch_size, X.shape[-1]))

        return X

    def _propagate(self, x, influences, **kwargs):
        """
        Propagate a single batch of examples through the network.

        First computes the activation the maps neurons, given a batch.
        Then updates the weights of the neurons by taking the mean of
        differences over the batch.

        :param X: an array of data
        :param influences: The influence at the current epoch,
        given the learning rate and map size
        :return: A vector describing activation values for each unit.
        """
        activation, difference_x = self.forward(x)
        self.backward(difference_x, influences, activation)

        return activation

    def forward(self, x, **kwargs):
        """
        Get the best matching units, based on euclidean distance.

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

        In this case (X - w) has been precomputed for speed, in the
        forward step.

        :param x: The difference vector between the weights and the input.
        :param influence: The influence the result has on each unit,
        depending on distance. Already includes the learning rate.
        """
        xp = cp.get_array_module(x)
        return xp.multiply(x, influence)

    def backward(self, diff_x, influences, activations, **kwargs):
        """
        Backward pass through the network, including update.

        :param diff_x: The difference between the input data and the weights
        :param influences: The influences at the current time-step
        :param activation: The activation at the output layer
        :param kwargs:
        :return: None
        """
        bmu = activations.__getattribute__(self.argfunc)(1)
        influence = influences[bmu]
        update = self._calculate_update(diff_x, influence)
        # If batch size is 1 we can leave out the call to mean.
        if update.shape[0] == 1:
            self.weights += update[0]
        else:
            u = update.mean(0)
            self.weights += update.mean(0)

    def distance_function(self, x, weights):
        """
        Calculate euclidean distance between a batch of input data and weights.

        :param x: The input
        :param weights: The weights
        :return: A tuple of matrices,
        The first matrix is a (batch_size * neurons) matrix of
        activation values, containing the response of each neuron
        to each input
        The second matrix is a (batch_size * neuron) matrix containing
        the difference between euch neuron and each input.
        """
        xp = cp.get_array_module(x)
        diff = self._distance_difference(x, weights)
        activations = xp.linalg.norm(diff, axis=2)

        return activations, diff

    def _distance_difference(self, x, weights):
        """
        Calculate the difference between an input and all the weights.

        :param x: The input.
        :param weights: An array of weights.
        :return: A vector of differences.
        """
        return x[:, None, :] - weights[None, :, :]

    def _calculate_influence(self, map_radius):
        """
        Pre-calculate the influence for a given value of sigma.

        The neighborhood has size map_dim * map_dim, so for a 30 * 30 map,
        the neighborhood will be size (900, 900).  d

        :param sigma: The neighborhood value.
        :return: The neighborhood
        """
        xp = cp.get_array_module(self.distance_grid)
        grid = xp.exp(-(self.distance_grid) / (2 * (map_radius ** 2)))
        return grid.reshape(self.weight_dim, self.weight_dim)[:, :, None]

    def _initialize_distance_grid(self):
        """
        Initialize the distance grid by calls to _grid_dist.

        :return:
        """
        p = [self._grid_distance(i) for i in range(self.weight_dim)]
        return np.array(p)

    def _grid_distance(self, index):
        """
        Calculate the distance grid for a single index position.

        This is pre-calculated for fast neighborhood calculations
        later on (see _calc_influence).

        :param index: The index for which to calculate the distances.
        :return: A flattened version of the distance array.
        """

        num_dim = len(self.map_dimensions)

        coord = []
        for idx, dim in enumerate(dimensions):
            if idx != 0:
                value = (index % dimensions[idx-1]) // dim
            else:
                value = index // dim
            coord.append(value)

        coord.append(index % self.map_dimensions[-1])

        for idx, (width, row) in enumerate(zip(self.map_dimensions, coord)):
            x = np.abs(np.arange(width) - row) ** 2
            dims = self.map_dimensions[::-1]
            if idx:
                dims = dims[:-idx]
            x = np.broadcast_to(x, dims).T
            if idx == 0:
                distance = np.copy(x)
            else:
                distance += x

        return distance

    def _check_input(self, X):
        """
        Check the input for validity.

        Ensures that the input data, X, is a 2-dimensional matrix, and that
        the second dimension of this matrix has the same dimensionality as
        the weight matrix.

        :param X: the input data
        :return: None
        """
        if X.ndim != 2:
            raise ValueError("Your data is not a 2D matrix. "
                             "Actual size: {0}".format(X.shape))

        if X.shape[1] != self.data_dimensionality:
            raise ValueError("Your data size != weight dim: {0}, "
                             "expected {1}".format(X.shape[1], self.data_dimensionality))

    def predict_distance(self, X, batch_size=100, show_progressbar=False):
        """
        Predict distances to some input data.

        This function should not be directly used.

        :param X: The input data.
        :return: A matrix, representing the activation
        each node has to each input.
        """
        self._check_input(X)

        xp = cp.get_array_module()
        batched = self._create_batches(X, batch_size, shuffle_data=False)

        activations = []

        for x in tqdm(batched, disable=not show_progressbar):
            activations.extend(self.forward(x)[0])

        activations = xp.asarray(activations, dtype=xp.float32)
        activations = activations[:X.shape[0]]
        return activations.reshape(X.shape[0], self.weight_dim)

    def predict(self, X, batch_size=100, show_progressbar=True):
        """
        Predict the BMU for each input data.

        :param X: Input data.
        :param batch_size: The batch size to use in prediction.
        :return: The index of the bmu which best describes the input data.
        """
        dist = self.predict_distance(X, batch_size, show_progressbar)
        res = dist.__getattribute__(self.argfunc)(1)
        xp = cp.get_array_module(res)
        if xp == np:
            return res
        else:
            return res.get()

    def quant_error(self, X, batch_size=1):
        """
        Calculate the quantization error.

        Find the the minimum euclidean distance between the units and
        some input.

        :param X: Input data.
        :param batch_size: The batch size to use when calculating
        the quantization error. Recommended to not raise this.
        :return: A vector of numbers, representing the quantization error
        for each data point.
        """
        dist = self.predict_distance(X, batch_size)
        res = dist.__getattribute__(self.valfunc)(1)
        xp = cp.get_array_module(res)
        if xp == np:
            return res
        else:
            return res.get()

    def topographic_error(self, X, batch_size=1):
        """
        Calculate the topographic error.

        The topographic error is a measure of the spatial organization of the
        map. Maps in which the most similar neurons are also close on the
        grid have low topographic error and indicate that a problem has been
        learned correctly.

        Formally, the topographic error is the proportion of units for which
        the two most similar neurons are not direct neighbors on the map.

        :param X: Input data.
        :param batch_size: The batch size to use when calculating
        the quantization error. Recommended to not raise this.
        :return: A vector of numbers, representing the topographic error
        for each data point.
        """
        dist = self.predict_distance(X, batch_size)
        xp = cp.get_array_module(dist)
        # Need to do a get here because cupy doesn't have argsort.
        if xp == cp:
            dist = dist.get()
        # Sort the distances and get the indices of the two smallest distances
        # for each datapoint.
        res = dist.argsort(1)[:, :2]
        # Lookup the euclidean distance between these points in the distance
        # grid
        dgrid = self.distance_grid.reshape(self.weight_dim, self.weight_dim)
        res = np.asarray([dgrid[x, y] for x, y in res])
        # Subtract 1.0 because 1.0 is the smallest distance.
        return np.sum(res > 1.0) / len(res)

    def neighbors(self):
        """Get all neighbors for all neurons."""
        dgrid = self.distance_grid.reshape(self.weight_dim, self.weight_dim)
        for x, y in zip(*np.nonzero(dgrid <= 2.0)):
            if x != y:
                yield x, y

    def neighbor_difference(self):
        """Get the euclidean distance between a node and its neighbors."""
        differences = np.zeros(self.weight_dim)
        num_neighbors = np.zeros(self.weight_dim)

        distance, _ = self.distance_function(self.weights, self.weights)
        for x, y in self.neighbors():
            differences[x] += distance[x, y]
            num_neighbors[x] += 1

        return differences / num_neighbors

    def spread(self, X):
        """Calculate the spread."""
        distance, _ = self.distance_function(X, self.weights)
        dists_per_node = defaultdict(list)
        for x, y in zip(np.argmin(distance, 1), distance):
            dists_per_node[x].append(y[x])

        out = np.zeros(self.weight_dim)
        average_spread = {k: np.mean(v)
                          for k, v in dists_per_node.items()}

        for x, y in average_spread.items():
            out[x] = y
        return out

    def receptive_field(self,
                        X,
                        identities,
                        max_len=10,
                        threshold=0.9,
                        batch_size=1):
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
        :param threshold: The threshold at which we consider a receptive field
        valid. If at least this proportion of the sequences of a neuron have
        the same suffix, that suffix is counted as acquired by the SOM.
        :param batch_size: The batch size to use in prediction
        :return: The receptive field of each neuron.
        """
        receptive_fields = defaultdict(list)
        predictions = self.predict(X, batch_size)

        if len(predictions) != len(identities):
            raise ValueError("X and identities are not the same length: "
                             "{0} and {1}".format(len(X), len(identities)))

        for idx, p in enumerate(predictions.tolist()):
            receptive_fields[p].append(identities[idx+1 - max_len:idx+1])

        sequence = defaultdict(list)

        for k, v in receptive_fields.items():

            v = [x for x in v if x]

            total = len(v)
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
        with the input which matches it the most, thus giving a good
        approximation of the "influence" of each input item.

        Works best for symbolic (instead of continuous) input data.

        parameters
        ==========
        X : numpy array
            Input data
        identities : list
            A list of names for each of the input data. Must be the same
            length as X.

        returns
        =======
        m : numpy array
            An array with the same shape as the map

        """
        xp = cp.get_array_module(X)

        if len(X) != len(identities):
            raise ValueError("X and identities are not the same length: "
                             "{0} and {1}".format(len(X), len(identities)))

        # Find all unique items in X
        X_unique, indices = np.unique(X, return_index=True, axis=0)

        node_match = []

        distances, _ = self.distance_function(X_unique, self.weights)

        if xp != np:
            distances = distances.get()

        for d in distances.argmin(0):
            node_match.append(identities[indices[d]])

        return np.array(node_match).reshape(self.map_dimensions)

    def map_weights(self):
        """
        Reshaped weights for visualization.

        The weights are reshaped as
        (W.shape[0], prod(W.shape[1:-1]), W.shape[2]).
        This allows one to easily see patterns, even for hyper-dimensional
        soms.

        For one-dimensional SOMs, the returned array is of shape
        (W.shape[0], 1, W.shape[2])

        returns
        =======
        w : numpy array
            A three-dimensional array containing the weights in a
            2D array for easy visualization.

        """
        first_dim = self.map_dimensions[0]
        if len(self.map_dimensions) != 1:
            second_dim = np.prod(self.map_dimensions[1:])
        else:
            second_dim = 1

        # Reshape to appropriate dimensions
        return self.weights.reshape((first_dim, second_dim, self.data_dimensionality))

    @classmethod
    def load(cls, path, array_type=np):
        """
        Load a SOM from a JSON file saved with this package.

        Note that it is necessary to specify which array library
        (i.e. cupy or numpy) you are using.

        parameters
        ==========
        path : str
            The path to the JSON file.
        array_type : library (i.e. numpy or cupy), optional, default numpy
            The array library to use.

        returns
        =======
        s : cls
            A som of the specified class.

        """
        data = json.load(open(path))

        weights = data['weights']
        weights = array_type.asarray(weights, dtype=array_type.float32)

        lrfunc = expo if data['lrfunc'] == 'expo' else linear
        nbfunc = expo if data['nbfunc'] == 'expo' else linear

        s = cls(data['map_dimensions'],
                data['data_dimensionality'],
                data['learning_rate'],
                lrfunc=lrfunc,
                nbfunc=nbfunc,
                neighborhood=data['neighborhood'],
                valfunc=data['valfunc'],
                argfunc=data['argfunc'])

        s.weights = weights
        s.trained = True

        return s

    def save(self, path):
        """Save a SOM to a JSON file."""
        to_save = {}
        for x in self.param_names:
            attr = self.__getattribute__(x)
            if type(attr) == np.ndarray or type(attr) == cp.ndarray:
                attr = [[float(x) for x in row] for row in attr]
            elif isinstance(attr, types.FunctionType):
                attr = attr.__name__
            to_save[x] = attr

        json.dump(to_save, open(path, 'w'))
