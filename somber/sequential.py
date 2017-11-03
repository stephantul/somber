import logging

import json
import cupy as cp
import numpy as np

from tqdm import tqdm
from .som import Som, shuffle
from .components.utilities import resize, Scaler
from .components.initializers import range_initialization
from functools import reduce

logger = logging.getLogger(__name__)


class Sequential(Som):
    """A base class for sequential SOMs, removing some code duplication."""

    def __init__(self,
                 map_dimensions,
                 data_dimensionality,
                 learning_rate,
                 neighborhood,
                 argfunc,
                 valfunc,
                 initializer,
                 scaler,
                 lr_lambda,
                 nb_lambda):

        super().__init__(map_dimensions,
                         data_dimensionality,
                         learning_rate,
                         neighborhood=neighborhood,
                         argfunc=argfunc,
                         valfunc=valfunc,
                         initializer=initializer,
                         scaler=scaler,
                         lr_lambda=lr_lambda,
                         nb_lambda=nb_lambda)

    def _ensure_params(self, X):
        """Ensure the parameters live on the GPU/CPU when the data does."""
        xp = cp.get_array_module(X)
        self.weights = xp.asarray(self.weights, xp.float32)
        self.context_weights = xp.asarray(self.context_weights, xp.float32)

    def _init_prev(self, X):
        """Initialize the context vector for recurrent SOMs."""
        xp = cp.get_array_module(X)
        return xp.zeros((X.shape[1], self.num_neurons))

    def _check_input(self, X):
        """
        Check the input for validity.

        Ensures that the input data, X, is a 3-dimensional matrix, and that
        the last dimension of this matrix has the same dimensionality as
        the weight matrix.
        """
        if X.ndim != 3:
            raise ValueError("Your data is not a 3D matrix. "
                             "Actual size: {0}".format(X.shape))

        if X.shape[-1] != self.data_dimensionality:
            raise ValueError("Your data size != weight dim: {0}, "
                             "expected {1}".format(X.shape[-1],
                                                   self.data_dimensionality))

    def _create_batches(self, X, batch_size, shuffle_data=True):
        """
        Create batches out of a sequence of data.

        This function will append zeros to the end of your data to ensure that
        all batches are even-sized. These are masked out during training.
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
        """Do a forward pass."""
        raise ValueError("Base class.")

    def predict_distance(self, X, batch_size=100, show_progressbar=False):
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
        return act.reshape(X_shape, self.num_neurons)

    def generate(self, num_to_generate, starting_place):
        """Generate data based on some initial position."""
        res = []
        activ = starting_place
        for x in range(num_to_generate):
            m = activ.__getattribute__(self.argfunc)(0)
            res.append(m)
            activ = self.context_weights[m]

        return res[::-1]


class Recursive(Sequential):
    """
    A recursive SOM.

    A recursive SOM models sequences through context dependence by not only
    storing the exemplars in weights, but also storing which exemplars
    preceded them. Because of this organization, the SOM can recursively
    "remember" short sequences, which makes it attractive for simple
    sequence problems, e.g. characters or words.

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
    context_weights : numpy or cupy array
        The weights which store the context dependence of the neurons.

    """

    param_names = {'data_dimensionality',
                   'learning_rate',
                   'lrfunc',
                   'map_dimensions',
                   'valfunc',
                   'argfunc'
                   'nbfunc',
                   'neighborhood',
                   'weights',
                   'context_weights',
                   'alpha',
                   'beta'}

    def __init__(self,
                 map_dimensions,
                 data_dimensionality,
                 alpha,
                 beta,
                 learning_rate,
                 neighborhood=None,
                 initializer=range_initialization,
                 scaler=Scaler(),
                 lr_lambda=2.5,
                 nb_lambda=2.5):

        super().__init__(map_dimensions,
                         data_dimensionality,
                         learning_rate,
                         neighborhood,
                         argfunc="argmax",
                         valfunc="max",
                         lr_lambda=lr_lambda,
                         nb_lambda=nb_lambda,
                         initializer=initializer,
                         scaler=scaler)

        self.context_weights = np.zeros((self.num_neurons, self.num_neurons),
                                        dtype=np.float32)
        self.alpha = alpha
        self.beta = beta

    def _propagate(self, x, influences, **kwargs):
        prev = kwargs['prev_activation']

        activation, diff_x, diff_y = self.forward(x, prev_activation=prev)
        x_update, y_update = self.backward(diff_x,
                                           influences,
                                           activation,
                                           diff_y=diff_y)
        # If batch size is 1 we can leave out the call to mean.
        if x_update.shape[0] == 1:
            self.weights += x_update[0]
        else:
            self.weights += x_update.mean(0)

        if y_update.shape[0] == 1:
            self.context_weights += y_update[0]
        else:
            self.context_weights += y_update.mean(0)

        return activation

    def forward(self, x, **kwargs):
        """
        Perform a forward pass through the network.

        The forward pass in recursive som is based on a combination between
        the activation in the last time-step and the current time-step.

        parameters
        ==========
        x : numpy or cupy array
            The input data.
        prev_activation : numpy or cupy array.
            The activation of the network in the previous time-step.

        returns
        =======
        activations : tuple of activations and differences
            A tuple containing the activation of each unit, the differences
            between the weights and input and the differences between the
            context input and context weights.

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
        differency_y : numpy or cupy array
            The differences between the input and context neurons.

        returns
        =======
        updates : tuple of arrays
            The updates to the weights and context weights, respectively.

        """
        xp = cp.get_array_module(diff_x)
        diff_y = kwargs['diff_y']
        bmu = activations.__getattribute__(self.argfunc)(1)
        influence = influence = influences[bmu]

        # Update
        x_update = xp.multiply(diff_x, influence)
        y_update = xp.multiply(diff_y, influence)

        return x_update, y_update

    @classmethod
    def load(cls, path, array_type=np):
        """
        Load a recursive SOM from a JSON file.

        You can use this function to load weights of other SOMs.
        If there are no context weights, they will be set to 0.

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
        weights = array_type.asarray(weights, dtype=cp.float32)

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
                data['data_dimensionality'],
                data['learning_rate'],
                neighborhood=data['neighborhood'],
                alpha=alpha,
                beta=beta,
                nb_lambda=data['nb_lambda'],
                lr_lambda=data['lr_lambda'])

        s.weights = weights
        s.context_weights = context_weights
        s.trained = True

        return s
