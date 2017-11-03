"""The standard SOM."""
import logging
import time
import types
import json
import cupy as cp
import numpy as np

from tqdm import tqdm
from .components.utilities import resize, Scaler, shuffle
from .components.initializers import range_initialization
from collections import Counter, defaultdict


logger = logging.getLogger(__name__)


class Som(object):
    """
    This is a batched version of the basic SOM.

    parameters
    ==========
    map_dimensions : tuple
        A tuple describing the map size. For example, (10, 10) will create
        a 10 * 10 map with 100 neurons, while a (10, 10, 10) map with 1000
        neurons creates a 10 * 10 * 10 map with 1000 neurons.
    data_dimensionality : int
        The dimensionality of the input data.
    learning_rate : float
        The starting learning rate h0.
    neighborhood : float, optional, default None.
        The starting neighborhood n0. If left at None, the value will be
        calculated as max(map_dimensions) / 2. This value might not be
        optimal for maps with more than 2 dimensions.
    argfunc : str, optional, default "argmin"
        The name of the function which is used for calculating the index of
        the BMU. This is necessary because we do not know in advance whether
        we will be receiving cupy or numpy arrays.
    valfunc : str, optional, default "min"
        The name of the function which is used for calculating the value of the
        BMU. This is necessary because we do not know in advance whether
        we will be receiving cupy or numpy arrays.
    initializer : function, optional, default range_initialization
        A function which takes in the input data and weight matrix and returns
        an initialized weight matrix. The initializers are defined in
        somber.components.initializers. Can be set to None.
    scaler : initialized Scaler instance, optional default Scaler()
        An initialized instance of Scaler() which is used to scale the data
        to have mean 0 and stdev 1.
    lr_lambda : float
        Controls the steepness of the exponential function that decreases
        the learning rate.
    nb_lambda : float
        Controls the steepness of the exponential function that decreases
        the neighborhood.

    attributes
    ==========
    trained : bool
        Whether the som has been trained.
    num_neurons : int
        The dimensionality of the weight matrix, i.e. the number of
        neurons on the map.
    distance_grid : numpy or cupy array
        An array which contains the distance from each neuron to each
        other neuron.

    """

    # Static property names
    param_names = {'neighborhood',
                   'learning_rate',
                   'map_dimensions',
                   'weights',
                   'data_dimensionality',
                   'lr_lambda',
                   'nb_lambda',
                   'valfunc',
                   'argfunc'}

    def __init__(self,
                 map_dimensions,
                 data_dimensionality,
                 learning_rate,
                 neighborhood=None,
                 argfunc="argmin",
                 valfunc="min",
                 initializer=range_initialization,
                 scaler=Scaler(),
                 lr_lambda=2.5,
                 nb_lambda=2.5):

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

        self.num_neurons = np.int(np.prod(map_dimensions))
        self.weights = np.zeros((self.num_neurons, data_dimensionality))
        self.data_dimensionality = data_dimensionality

        # Initialize the distance grid: only needs to be done once.
        self.distance_grid = self._initialize_distance_grid()
        self.argfunc = argfunc
        self.valfunc = valfunc
        self.trained = False
        self.scaler = scaler
        self.initializer = initializer
        self.nb_lambda = nb_lambda
        self.lr_lambda = lr_lambda

    def fit(self,
            X,
            num_epochs=10,
            updates_epoch=10,
            stop_lr_updates=None,
            stop_nb_updates=None,
            batch_size=1,
            show_progressbar=False):
        """
        Fit the SOM to some data.

        parameters
        ==========
        X : numpy or cupy array.
            The input data.
        num_epochs : int, optional, default 10
            The number of epochs to train for.
        updates_epoch : int, optional, default 10
            The number of updates to perform on the learning rate and
            neighborhood per epoch. 10 suffices for most problems.
        stop_lr_updates : int, optional, default None
            The epoch at which to stop updating the learning rate.
            If this is set to None, the learning rate is always updated.
        stop_nb_updates : float, optional, default None
            The epoch at which to stop updating the neighborhood
        batch_size : int, optional, default 100
            The batch size to use. Warning: batching can change your
            performance dramatically, depending on the task.

        """
        xp = cp.get_array_module(X)
        X = xp.asarray(X, dtype=xp.float32)
        self._check_input(X)

        X = self.scaler.fit_transform(X)

        if self.initializer is not None:
            self.weights = self.initializer(X, self.weights)

        self._ensure_params(X)
        start = time.time()

        if stop_lr_updates is None:
            stop_lr_updates = num_epochs
        if stop_nb_updates is None:
            stop_nb_updates = num_epochs

        total_lr_updates = stop_lr_updates * updates_epoch
        total_nb_updates = stop_nb_updates * updates_epoch

        one_step = np.exp(-((1.0 - (1.0 / total_nb_updates))) * self.nb_lambda)
        nb_constant = np.exp(-(1.0 * self.nb_lambda)) / one_step

        one_step = np.exp(-((1.0 - (1.0 / total_lr_updates))) * self.lr_lambda)
        lr_constant = np.exp(-(1.0 * self.lr_lambda)) / one_step

        for epoch in range(num_epochs):

            logger.info("Epoch {0} of {1}".format(epoch, num_epochs))

            self._epoch(X,
                        epoch,
                        batch_size,
                        updates_epoch,
                        nb_constant,
                        lr_constant,
                        show_progressbar)

        self.trained = True
        self.weights = self.scaler.inverse_transform(self.weights)

        logger.info("Total train time: {0}".format(time.time() - start))

    def _epoch(self,
               X,
               epoch_idx,
               batch_size,
               updates_epoch,
               nb_constant,
               lr_constant,
               show_progressbar):
        """
        Run a single epoch.

        This function shuffles the data internally,
        as this improves performance.

        parameters
        ==========
        X : numpy or cupy array
            The training data.
        epoch_idx : int
            The current epoch
        batch_size : int
            The batch size
        updates_epoch : int
            The number of updates to perform per epoch
        nb_constant : float
            The number to multiply the neighborhood with at every update step.
        lr_constant : float
            The number to multiply the learning rate with at every update step.
        show_progressbar : bool
            Whether to show a progressbar during training.

        """
        updates_so_far = epoch_idx * updates_epoch

        # Create batches
        X_ = self._create_batches(X, batch_size)
        X_len = np.prod(X.shape[:-1])

        map_radius = self.neighborhood * (nb_constant ** updates_so_far)
        influences = self._calculate_influence(map_radius)

        lr_decay = (lr_constant ** updates_so_far)
        learning_rate = self.learning_rate * lr_decay

        influences *= learning_rate

        update_step = np.ceil(X.shape[0] / updates_epoch)

        # Initialize the previous activation
        prev_activation = self._init_prev(X_)

        # Iterate over the training data
        for idx, x in enumerate(tqdm(X_, disable=not show_progressbar)):

            # Our batches are padded, so we need to
            # make sure we know when we hit the padding
            # so we don't inadvertently learn zeroes.
            diff = X_len - (idx * batch_size)
            if diff and diff < batch_size:
                x = x[:diff]

            if idx % update_step == 0:

                prev_lr = learning_rate
                learning_rate *= lr_constant
                logger.info("Updated learning rate: {0}".format(learning_rate))
                # Recalculate the influences given learning rate
                influences *= (learning_rate / prev_lr)

            if idx % update_step == 0:

                logger.info("Updated map radius: {0}".format(map_radius))
                map_radius *= nb_constant
                influences = self._calculate_influence(map_radius)
                influences *= learning_rate

            prev_activation = self._propagate(x,
                                              influences,
                                              prev_activation=prev_activation)

    def _ensure_params(self, X):
        """Ensure the parameters live on the GPU/CPU when the data does."""
        xp = cp.get_array_module(X)
        self.weights = xp.asarray(self.weights, xp.float32)
        self.distance_grid = xp.asarray(self.distance_grid, xp.int32)

    def _init_prev(self, x):
        """Initialize recurrent SOMs."""
        return None

    def _create_batches(self, X, batch_size, shuffle_data=True):
        """
        Create batches out of a sequence of data.

        This function will append zeros to the end of your data to ensure that
        all batches are even-sized. These are masked out during training.
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
        """Propagate a single batch of examples through the network."""
        activation, difference_x = self.forward(x)
        update = self.backward(difference_x, influences, activation)
        # If batch size is 1 we can leave out the call to mean.
        if update.shape[0] == 1:
            self.weights += update[0]
        else:
            self.weights += update.mean(0)

        return activation

    def forward(self, x, **kwargs):
        """
        Get the best matching neurons, and the difference between inputs.

        Note: it might seem like this function can be replaced by a call to
        distance function. This is only true for the regular SOM. other
        SOMs, like the recurrent SOM need more complicated forward pass
        functions.

        parameters
        ==========
        x : numpy or cupy array.
            The input vector.

        returns
        =======
        matrices : tuple of matrices.
            A tuple containing the activations and differences between
            neurons and input, respectively.

        """
        return self.distance_function(x, self.weights)

    def backward(self, diff_x, influences, activations, **kwargs):
        """
        Backward pass through the network, including update.

        parameters
        ==========
        diff_x : numpy or cupy array
            A matrix containing the differences between the input and neurons.
        influences : numpy or cupy array
            A matrix containing the influence each neuron has on each
            other neuron. This is used to calculate the updates.
        activations : numpy or cupy array
            The activations each neuron has to each data point. This is used
            to calculate the BMU.

        returns
        =======
        update : numpy or cupy array
            A numpy array containing the updates to the neurons.

        """
        xp = cp.get_array_module(diff_x)

        bmu = activations.__getattribute__(self.argfunc)(1)
        influence = influences[bmu]
        update = xp.multiply(diff_x, influence)
        return update

    def distance_function(self, x, weights):
        """
        Calculate euclidean distance between a batch of input data and weights.

        parameters
        ==========
        X : numpy or cupy array.
            The input data.
        weights : numpy or cupy array.
            The input data.

        returns
        =======
        matrices : tuple of matrices
            The first matrix is a (batch_size * neurons) matrix of
            activation values, containing the response of each neuron
            to each input
            The second matrix is a (batch_size * neurons) matrix containing
            the difference between euch neuron and each input.

        """
        xp = cp.get_array_module(x)
        diff = x[:, None, :] - weights[None, :, :]
        activations = xp.linalg.norm(diff, axis=2)

        return activations, diff

    def _calculate_influence(self, neighborhood):
        """
        Pre-calculate the influence for a given value of sigma.

        The neighborhood has size num_neurons * num_neurons, so for a
        30 * 30 map, the neighborhood will be size (900, 900).

        parameters
        ==========
        neighborhood : float
            The neighborhood value.

        returns
        =======
        neighborhood : numpy array
            The influence from each neuron to each other neuron.

        """
        xp = cp.get_array_module(self.distance_grid)
        grid = xp.exp(-(self.distance_grid) / (2 * (neighborhood ** 2)))
        return grid.reshape(self.num_neurons, self.num_neurons)[:, :, None]

    def _initialize_distance_grid(self):
        """Initialize the distance grid by calls to _grid_dist."""
        p = [self._grid_distance(i) for i in range(self.num_neurons)]
        return np.array(p)

    def _grid_distance(self, index):
        """
        Calculate the distance grid for a single index position.

        This is pre-calculated for fast neighborhood calculations
        later on (see _calc_influence).
        """
        # Take every dimension but the first in reverse
        # then reverse that list again.
        dimensions = np.cumprod(self.map_dimensions[1::][::-1])[::-1]

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
                distance += x.T

        return distance

    def _check_input(self, X):
        """
        Check the input for validity.

        Ensures that the input data, X, is a 2-dimensional matrix, and that
        the second dimension of this matrix has the same dimensionality as
        the weight matrix.
        """
        if X.ndim != 2:
            raise ValueError("Your data is not a 2D matrix. "
                             "Actual size: {0}".format(X.shape))

        if X.shape[1] != self.data_dimensionality:
            raise ValueError("Your data size != weight dim: {0}, "
                             "expected {1}".format(X.shape[1],
                                                   self.data_dimensionality))

    def predict_distance(self, X, batch_size=100, show_progressbar=False):
        """
        Predict distances to some input data.

        parameters
        ==========
        X : numpy or cupy array.
            The input data.
        batch_size : int, optional, default 100
            The batch size to use in prediction. This may affect prediction
            in stateful, i.e. sequential SOMs.
        show_progressbar : bool
            Whether to show a progressbar during prediction.

        returns
        =======
        predictions : numpy array
            A matrix containing the distance from each datapoint to all
            neurons. The distance is normally expressed as euclidean distance,
            but can be any arbitrary metric.

        """
        self._check_input(X)

        xp = cp.get_array_module()
        batched = self._create_batches(X, batch_size, shuffle_data=False)

        activations = []

        for x in tqdm(batched, disable=not show_progressbar):
            activations.extend(self.forward(x)[0])

        activations = xp.asarray(activations, dtype=xp.float32)
        activations = activations[:X.shape[0]]
        return activations.reshape(X.shape[0], self.num_neurons)

    def predict(self, X, batch_size=100, show_progressbar=True):
        """
        Predict the BMU for each input data.

        parameters
        ==========
        X : numpy or cupy array.
            The input data.
        batch_size : int, optional, default 100
            The batch size to use in prediction. This may affect prediction
            in stateful, i.e. sequential SOMs.
        show_progressbar : bool
            Whether to show a progressbar during prediction.

        returns
        =======
        predictions : numpy array
            An array containing the BMU for each input data point.

        """
        dist = self.predict_distance(X, batch_size, show_progressbar)
        res = dist.__getattribute__(self.argfunc)(1)
        xp = cp.get_array_module(res)
        if xp == np:
            return res
        else:
            return res.get()

    def quantization_error(self, X, batch_size=1):
        """
        Calculate the quantization error.

        Find the the minimum euclidean distance between the units and
        some input.

        parameters
        ==========
        X : numpy or cupy array.
            The input data.
        batch_size : int
            The batch size to use for processing.

        returns
        =======
        error : numpy array
            The error for each data point.

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

        parameters
        ==========
        X : numpy or cupy array.
            The input data.
        batch_size : int
            The batch size to use when calculating the topographic error.

        returns
        =======
        error : numpy array
            A vector of numbers, representing the topographic error
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
        dgrid = self.distance_grid.reshape(self.num_neurons, self.num_neurons)
        res = np.asarray([dgrid[x, y] for x, y in res])
        # Subtract 1.0 because 1.0 is the smallest distance.
        return np.sum(res > 1.0) / len(res)

    def neighbors(self):
        """Get all neighbors for all neurons."""
        dgrid = self.distance_grid.reshape(self.num_neurons, self.num_neurons)
        for x, y in zip(*np.nonzero(dgrid <= 2.0)):
            if x != y:
                yield x, y

    def neighbor_difference(self):
        """Get the euclidean distance between a node and its neighbors."""
        differences = np.zeros(self.num_neurons)
        num_neighbors = np.zeros(self.num_neurons)

        distance, _ = self.distance_function(self.weights, self.weights)
        for x, y in self.neighbors():
            differences[x] += distance[x, y]
            num_neighbors[x] += 1

        return differences / num_neighbors

    def spread(self, X):
        """
        Calculate the average spread for each node.

        The average spread is a measure of how far each neuron is from the
        data points which cluster to it.

        parameters
        ==========
        X : numpy array or cupy array
            The input data.

        returns
        =======
        spread : numpy array
            The average distance from each neuron to each data point.

        """
        distance, _ = self.distance_function(X, self.weights)
        dists_per_neuron = defaultdict(list)
        for x, y in zip(np.argmin(distance, 1), distance):
            dists_per_neuron[x].append(y[x])

        out = np.zeros(self.num_neurons)
        average_spread = {k: np.mean(v)
                          for k, v in dists_per_neuron.items()}

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

        parameters
        ==========
        X : numpy or cupy array
            Input data.
        identities : list
            A list of symbolic identities associated with each input.
            We expect this list to be as long as the input data.
        max_len : int, optional, default 10
            The maximum length to attempt to find. Raising this increases
            memory use.
        threshold : float, optional, default .9
            The threshold at which we consider a receptive field
            valid. If at least this proportion of the sequences of a neuron
            have the same suffix, that suffix is counted as acquired by the
            SOM.
        batch_size : int, optional, default 1
            The batch size to use in prediction

        returns
        =======
        receptive_fields : dict
            A dictionary mapping from the neuron id to the found sequences
            for that neuron. The sequences are represented as lists of
            symbols from identities.

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
        return self.weights.reshape((first_dim,
                                     second_dim,
                                     self.data_dimensionality))

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

        s = cls(data['map_dimensions'],
                data['data_dimensionality'],
                data['learning_rate'],
                neighborhood=data['neighborhood'],
                valfunc=data['valfunc'],
                argfunc=data['argfunc'],
                nb_lambda=data['nb_lambda'],
                lr_lambda=data['lr_lambda'])

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
