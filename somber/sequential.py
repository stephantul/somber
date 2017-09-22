import logging

import json
import cupy as cp
import numpy as np

from tqdm import tqdm
from .som import Som, shuffle
from .components.utilities import expo, linear, resize
from functools import reduce

logger = logging.getLogger(__name__)


class Sequential(Som):
    """
    A base class for sequential SOMs, removing some code duplication.

    Not usable stand-alone.
    """

    def __init__(self,
                 map_dim,
                 data_dim,
                 learning_rate,
                 sigma,
                 lrfunc=expo,
                 nbfunc=expo,
                 min_max="none"):
        """
        A base class for sequential SOMs, removing some code duplication.

        :param map_dim: A tuple describing the MAP size.
        :param data_dim: The dimensionality of the input matrix.
        :param learning_rate: The learning rate.
        :param sigma: The neighborhood factor.
        :param lrfunc: The function used to decrease the learning rate.
        :param nbfunc: The function used to decrease the neighborhood
        :param min_max: The function used to determine the winner.
        """
        super().__init__(map_dim,
                         data_dim,
                         learning_rate,
                         sigma,
                         lrfunc,
                         nbfunc,
                         min_max)

    def _ensure_params(self, X):
        """
        Ensure the parameters are of the correct type.

        :param X: The input data
        :return: None
        """
        xp = cp.get_array_module(X)
        self.weights = xp.asarray(self.weights, xp.float32)
        self.context_weights = xp.asarray(self.context_weights, xp.float32)

    def _init_prev(self, X):
        """
        Safely initializes the first previous activation.

        :param X: The input data.
        :return: A matrix of the appropriate size for simulating contexts.
        """
        xp = cp.get_array_module(X)
        return xp.zeros((X.shape[1], self.weight_dim))

    def _check_input(self, X):
        """
        Check the input for validity.

        Ensures that the input data, X, is a 2-dimensional matrix, and that
        the second dimension of this matrix has the same dimensionality as
        the weight matrix.

        :param X: the input data
        :return: None
        """
        if X.ndim != 3:
            raise ValueError("Your data is not a 3D matrix. "
                             "Actual size: {0}".format(X.shape))

        if X.shape[-1] != self.data_dim:
            raise ValueError("Your data size != weight dim: {0}, "
                             "expected {1}".format(X.shape[-1], self.data_dim))

    def _create_batches(self, X, batch_size, shuffle_data=True):
        """
        Create subsequences out of a sequential piece of data.

        Assumes ndim(X) == 3.

        This function will append zeros to the end of your data to make
        sure all batches even-sized.

        :param X: A numpy array, representing your input data.
        Must have 3 dimensions.
        :param batch_size: The desired batch size.
        :return: A batched version of your data
        """
        xp = cp.get_array_module(X)

        # Total length of sequence in items
        sequence_length = X.shape[0] * X.shape[1]

        if shuffle_data:
            X = shuffle(X)

        if batch_size > sequence_length:
            batch_size = sequence_length

        max_x = int(xp.ceil(sequence_length / batch_size))
        # This line first resizes the data to
        X = resize(X, (batch_size, max_x, X.shape[2]))
        # Transposes it to (len(X) / batch_size, batch_size, data_dim)

        return X.transpose((1, 0, 2))

    def forward(self, x, **kwargs):
        """
        Empty.

        :param x: the input data
        :return None
        """
        pass

    def _predict_base(self, X, batch_size=100, show_progressbar=False):
        """
        Predict distances to some input data.

        :param X: The input data.
        :return: An array of arrays, representing the activation
        each node has to each input.
        """
        self._check_input(X)

        X_shape = reduce(np.multiply, X.shape[:-1], 1)

        xp = cp.get_array_module(X)
        batched = self._create_batches(X, batch_size)

        activations = []

        activation = self._init_prev(batched)

        for x in tqdm(batched, disable=not show_progressbar):
            activation = self.forward(x, prev_activation=activation)[0]
            activations.append(activation)

        act = xp.asarray(activations, dtype=xp.float32).transpose((1, 0, 2))
        act = act[:X_shape]
        return act.reshape(X_shape, self.weight_dim)

    def generate(self, num_to_generate, starting_place):
        """
        Generate data based on some initial position.

        :param num_to_generate: The number of tokens to generate
        :param starting_place: The place to start from. This should
        be a vector equal to a context weight.
        :return:
        """
        res = []
        activ = starting_place
        for x in range(num_to_generate):
            m = np.argmax(activ, 0)
            res.append(m)
            activ = self.context_weights[m]

        return res[::-1]


class Recursive(Sequential):

    param_names = {'data_dim',
                   'learning_rate',
                   'lrfunc',
                   'map_dimensions',
                   'min_max',
                   'nbfunc',
                   'sigma',
                   'weights',
                   'context_weights',
                   'alpha',
                   'beta'}

    def __init__(self,
                 map_dim,
                 data_dim,
                 learning_rate,
                 alpha,
                 beta,
                 sigma=None,
                 lrfunc=expo,
                 nbfunc=expo):
        """
        A recursive SOM.

        A recursive SOM models sequences through context dependence by not only
        storing the exemplars in weights, but also storing which exemplars
        preceded them. Because of this organization, the SOM can recursively
        "remember" short sequences, which makes it attractive for simple
        sequence problems, e.g. characters or words.

        :param map_dim: A tuple of map dimensions,
        e.g. (10, 10) instantiates a 10 by 10 map.
        :param data_dim: The data dimensionality.
        :param learning_rate: The learning rate, which is decreased
        according to some function.
        :param lrfunc: The function to use in decreasing the learning rate.
        The functions are defined in utils. Default is exponential.
        :param nbfunc: The function to use in decreasing the neighborhood size.
        The functions are defined in utils. Default is exponential.
        :param alpha: a float value, specifying how much weight the
        input value receives in the BMU calculation.
        :param beta: a float value, specifying how much weight the context
        receives in the BMU calculation.
        :param sigma: The starting value for the neighborhood size, which is
        decreased over time. If sigma is None (default), sigma is calculated as
        ((max(map_dim) / 2) + 0.01), which is generally a good value.
        """

        super().__init__(map_dim,
                         data_dim,
                         learning_rate,
                         lrfunc,
                         nbfunc,
                         sigma,
                         min_max=np_max)

        self.context_weights = np.zeros((self.weight_dim, self.weight_dim),
                                        dtype=np.float32)
        self.alpha = alpha
        self.beta = beta

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
        prev = kwargs['prev_activation']

        activation, diff_x, diff_y = self.forward(x, prev_activation=prev)
        self.backward(diff_x, influences, activation, difference_y=diff_y)
        return activation

    def forward(self, x, **kwargs):
        """
        Get the best matching units, based on euclidean distance.

        The euclidean distance between the context vector and context weights
        and input vector and weights are used to estimate the BMU. The
        activation of the units is the sum of the distances, weighed by two
        constants, alpha and beta.

        The exponent of the negative of this value describes the activation
        of the units. This function is bounded between 0 and 1, where 1 means
        the unit matches well and 0 means the unit doesn't match at all.

        :param x: A batch of data.
        :return: The activation, the difference between the input and weights.
        and the difference between the context and weights.
        """
        prev = kwargs['prev_activation']
        xp = cp.get_array_module(self.weights)
        # Differences is the components of the weights subtracted from
        # the weight vector.
        distance_x, diff_x = self.distance_function(x, self.weights)
        distance_y, diff_y = self.distance_function(prev, self.context_weights)

        x_ = distance_x * self.alpha
        y_ = distance_y * self.beta
        activation = xp.exp(-(x_ + y_))

        return activation, diff_x, diff_y

    def backward(self, diff_x, influences, activation, **kwargs):
        """
        Backward pass through the network, including update.

        :param diff_x: The difference between the input data and the weights
        :param influences: The influences at the current time-step
        :param activation: The activation at the output
        :param kwargs:
        :return: None
        """
        xp = cp.get_array_module(diff_x)

        diff_y = kwargs['difference_y']
        influence = self._apply_influences(activation, influences)
        # Update
        x_update = self._calculate_update(diff_x, influence)
        self.weights += x_update.mean(0)
        y_update = self._calculate_update(diff_y, influence)
        self.context_weights += xp.squeeze(y_update.mean(0))

    @classmethod
    def load(cls, path, array_type=np):
        """
        Load a recursive SOM from a JSON file.

        You can use this function to load weights of other SOMs.
        If there are no context weights, the context weights will be set to 0.

        :param path: The path to the JSON file.
        :param array_type: The type of array to load
        :return: A RecSOM.
        """
        data = json.load(open(path))

        weights = data['weights']
        weights = array_type.asarray(weights, dtype=cp.float32)
        lrfunc = expo if data['lrfunc'] == 'expo' else linear
        nbfunc = expo if data['nbfunc'] == 'expo' else linear
        min_max = np_min if data['min_max'] == 'np_min' else np_max

        try:
            context_weights = data['context_weights']
            context_weights = array_type.asarray(context_weights,
                                                 dtype=cp.float32)
        except KeyError:
            context_weights = array_type.zeros((len(weights), len(weights)))

        try:
            alpha = data['alpha']
            beta = data['beta']
        except KeyError:
            alpha = 1.0
            beta = 1.0

        s = cls(data['map_dimensions'],
                data['data_dim'],
                data['learning_rate'],
                lrfunc=lrfunc,
                nbfunc=nbfunc,
                sigma=data['sigma'],
                alpha=alpha,
                beta=beta)

        s.weights = weights
        s.context_weights = context_weights
        s.trained = True

        return s
