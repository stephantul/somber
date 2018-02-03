"""The standard SOM."""
import logging
import json
import numpy as np

from .components.initializers import range_initialization
from collections import Counter, defaultdict
from .base import Base


logger = logging.getLogger(__name__)


class Som(Base):
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
        the BMU.
    valfunc : str, optional, default "min"
        The name of the function which is used for calculating the value of the
        BMU.
    initializer : function, optional, default range_initialization
        A function which takes in the input data and weight matrix and returns
        an initialized weight matrix. The initializers are defined in
        somber.components.initializers. Can be set to None.
    scaler : initialized Scaler instance, optional default None
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
    distance_grid : numpy array
        An array which contains the distance from each neuron to each
        other neuron.

    """

    # Static property names
    param_names = {'map_dimensions',
                   'weights',
                   'data_dimensionality',
                   'params'}

    def __init__(self,
                 map_dimensions,
                 data_dimensionality,
                 learning_rate,
                 influence=None,
                 initializer=range_initialization,
                 scaler=None,
                 lr_lambda=2.5,
                 infl_lambda=2.5):
        """Organize your maps."""
        if influence is None:
            # Add small constant to sigma to prevent
            # divide by zero for maps with the same max_dim as the number
            # of dimensions.
            influence = max(map_dimensions) / 2
            influence += 0.0001

        # A tuple of dimensions
        # Usually (width, height), but can accomodate N-dimensional maps.
        self.map_dimensions = map_dimensions

        num_neurons = np.int(np.prod(map_dimensions))
        params = {'infl': {'value': influence,
                           'factor': infl_lambda,
                           'orig': influence},
                  'lr': {'value': learning_rate,
                         'factor': lr_lambda,
                         'orig': learning_rate}}

        super().__init__(num_neurons,
                         data_dimensionality,
                         params,
                         'argmin',
                         'min',
                         initializer,
                         scaler)
        # Initialize the distance grid: only needs to be done once.
        self.distance_grid = self._initialize_distance_grid()

    def _init_prev(self, x):
        """Initialize recurrent SOMs."""
        return None

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
        grid = np.exp(-(self.distance_grid) / (2 * (neighborhood ** 2)))
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
        X : numpy array.
            The input data.
        batch_size : int
            The batch size to use when calculating the topographic error.

        returns
        =======
        error : numpy array
            A vector of numbers, representing the topographic error
            for each data point.

        """
        dist = self.transform(X, batch_size)
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
        X : numpy array
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
        X : numpy array
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

        rec = {}

        for k, v in receptive_fields.items():
            # if there's only one sequence, we don't know
            # anything abouw how salient it is.
            seq = []
            if len(v) <= 1:
                continue
            else:
                for x in reversed(list(zip(*v))):
                    x = Counter(x)
                    if x.most_common(1)[0][1] / sum(x.values()) > threshold:
                        seq.append(x.most_common(1)[0][0])
                    else:
                        rec[k] = seq
                        break

        return rec

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
        distances = self.transform(X)

        if len(distances) != len(identities):
            raise ValueError("X and identities are not the same length: "
                             "{0} and {1}".format(len(X), len(identities)))

        node_match = []

        for d in distances.__getattribute__(self.argfunc)(0):
            node_match.append(identities[d])

        return np.array(node_match)

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
    def load(cls, path):
        """
        Load a SOM from a JSON file saved with this package..

        parameters
        ==========
        path : str
            The path to the JSON file.

        returns
        =======
        s : cls
            A som of the specified class.

        """
        data = json.load(open(path))

        weights = data['weights']
        weights = np.asarray(weights, dtype=np.float32)

        s = cls(data['map_dimensions'],
                data['data_dimensionality'],
                data['params']['lr']['orig'],
                influence=data['params']['infl']['orig'],
                lr_lambda=data['params']['lr']['factor'],
                infl_lambda=data['params']['infl']['factor'])

        s.weights = weights
        s.trained = True

        return s
