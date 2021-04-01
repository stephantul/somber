"""The sequential SOMs."""
import json
import logging
from functools import reduce
from typing import Callable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from somber.components.utilities import shuffle
from somber.components.initializers import range_initialization
from somber.ng import Ng
from somber.som import Som
from somber.base import _T


logger = logging.getLogger(__name__)


class SequentialMixin(object):
    """A base class for sequential SOMs, removing some code duplication."""

    def _init_prev(self, X: np.ndarray) -> np.ndarray:
        """Initialize the context vector for recurrent SOMs."""
        return np.zeros((X.shape[1], self.num_neurons))

    def _create_batches(
        self, X: np.ndarray, batch_size: int, shuffle_data: bool = False
    ) -> np.ndarray:
        """
        Create batches out of a sequence of data.

        This function will append zeros to the end of your data to ensure that
        all batches are even-sized. These are masked out during training.
        """
        if shuffle_data:
            X = shuffle(X)

        if batch_size > X.shape[0]:
            batch_size = X.shape[0]

        max_x = int(np.ceil(X.shape[0] / batch_size))
        # This line first resizes the data to
        X = np.resize(X, (batch_size, max_x, X.shape[1]))
        # Transposes it to (len(X) / batch_size, batch_size, data_dim)
        return X.transpose((1, 0, 2))

    def predict_distance(
        self, X: np.ndarray, batch_size: int = 1, show_progressbar: bool = False
    ) -> np.ndarray:
        """Predict distances to some input data."""
        X = self._check_input(X)

        X_shape = reduce(np.multiply, X.shape[:-1], 1)

        batched = self._create_batches(X, batch_size, shuffle_data=False)

        activations = []

        activation = self._init_prev(batched)

        for x in tqdm(batched, disable=not show_progressbar):
            activation = self.forward(x, prev_activation=activation)[0]
            activations.append(activation)

        act = np.asarray(activations, dtype=np.float64).transpose((1, 0, 2))
        act = act[:X_shape]
        return act.reshape(X_shape, self.num_neurons)

    def generate(self, num_to_generate: int, starting_place: np.ndarray) -> List[int]:
        """Generate data based on some initial position."""
        res = []
        activ = starting_place[None, :]
        index = activ.__getattribute__(self.argfunc)(1)
        item = self.weights[index]
        for x in range(num_to_generate):
            activ = self.forward(item, prev_activation=activ)[0]
            index = activ.__getattribute__(self.argfunc)(1)
            res.append(index)
            item = self.weights[index]

        return res


class RecursiveMixin(SequentialMixin):

    param_names = {
        "data_dimensionality",
        "params",
        "map_dimensions",
        "valfunc",
        "argfunc",
        "weights",
        "context_weights",
        "alpha",
        "beta",
    }

    def _propagate(
        self, x: np.ndarray, influences: np.ndarray, prev_activation: np.ndarray
    ) -> np.ndarray:
        activation, diff_x, diff_y = self.forward(x, prev_activation=prev_activation)
        x_update, y_update = self.backward(
            diff_x, influences, activation, diff_y=diff_y
        )
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

    def forward(
        self, x: np.ndarray, prev_activation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a forward pass through the network.

        The forward pass in recursive som is based on a combination between
        the activation in the last time-step and the current time-step.

        :param x: The input data.
        :param prev_activation: The activation of the network in the previous time-step.
        :return: A tuple containing the activation of each unit, the differences
            between the weights and input and the differences between the
            context input and context weights.
        """
        # Differences is the components of the weights subtracted from
        # the weight vector.
        distance_x, diff_x = self.distance_function(x, self.weights)
        distance_y, diff_y = self.distance_function(
            prev_activation, self.context_weights
        )

        x_ = distance_x * self.alpha
        y_ = distance_y * self.beta
        activation = np.exp(-(x_ + y_))

        return activation, diff_x, diff_y

    @classmethod
    def load(cls, path: str) -> _T:
        """
        Load a recursive SOM from a JSON file.

        You can use this function to load weights of other SOMs.
        If there are no context weights, they will be set to 0.

        :param path: The path to the JSON file.
        :param return: A som of the specified class
        """
        data = json.load(open(path))

        weights = data["weights"]
        weights = np.asarray(weights, dtype=np.float64)

        try:
            context_weights = data["context_weights"]
            context_weights = np.asarray(context_weights, dtype=np.float64)
        except KeyError:
            context_weights = np.zeros((len(weights), len(weights)))

        try:
            alpha = data["alpha"]
            beta = data["beta"]
        except KeyError:
            alpha = 1.0
            beta = 1.0

        s = cls(
            data["map_dimensions"],
            data["data_dimensionality"],
            data["params"]["lr"]["orig"],
            influence=data["params"]["infl"]["orig"],
            alpha=alpha,
            beta=beta,
            lr_lambda=data["params"]["lr"]["factor"],
            infl_lambda=data["params"]["infl"]["factor"],
        )

        s.weights = weights
        s.context_weights = context_weights
        s.trained = True

        return s


class RecursiveSom(RecursiveMixin, Som):
    """Recursive version of the SOM."""

    def __init__(
        self,
        map_dimensions: Tuple[int],
        learning_rate: float,
        alpha: float,
        beta: float,
        data_dimensionality: Optional[int] = None,
        influence: Optional[float] = None,
        initializer: Optional[Callable] = range_initialization,
        lr_lambda: float = 2.5,
        infl_lambda: float = 2.5,
    ) -> None:
        """Organize your maps recursively."""
        super().__init__(
            map_dimensions,
            learning_rate,
            data_dimensionality,
            influence,
            initializer,
            lr_lambda,
            infl_lambda,
        )

        self.alpha = alpha
        self.beta = beta
        self.argfunc = "argmax"
        self.valfunc = "max"

        self.context_weights = np.zeros(
            (self.num_neurons, self.num_neurons), dtype=np.float64
        )

    def backward(self, diff_x, influences, activations, diff_y) -> np.ndarray:
        """
        Backward pass through the network, including update.

        :param diff_x: A matrix containing the differences between the input and neurons.
        :param influences: A matrix containing the influence each neuron has on each
            other neuron. This is used to calculate the updates.
        :param activations: The activations each neuron has to each data point. This is used
            to calculate the BMU.
        :param diff_y: The differences between the input and context neurons.
        :return: The updates to the weights and context weights, respectively.
        """
        bmu = self._get_bmu(activations)
        influence = influences[bmu]

        # Update
        x_update = np.multiply(diff_x, influence)
        y_update = np.multiply(diff_y, influence)

        return x_update, y_update


class RecursiveNg(RecursiveMixin, Ng):
    """Recursive version of the neural gas."""

    def __init__(
        self,
        num_neurons: int,
        data_dimensionality: int,
        learning_rate: float,
        alpha: float,
        beta: float,
        influence: float,
        initializer: Optional[Callable] = range_initialization,
        lr_lambda: float = 2.5,
        infl_lambda: float = 2.5,
    ):
        """Organize your gas recursively."""
        super().__init__(
            num_neurons,
            data_dimensionality,
            learning_rate,
            influence,
            initializer,
            lr_lambda,
            infl_lambda,
        )

        self.alpha = alpha
        self.beta = beta
        self.argfunc = "argmax"
        self.valfunc = "max"

        self.context_weights = np.zeros(
            (self.num_neurons, self.num_neurons), dtype=np.float64
        )

    def backward(
        self,
        diff_x: np.ndarray,
        influences: np.ndarray,
        activations: np.ndarray,
        diff_y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass through the network, including update.

        :param diff_x: A matrix containing the differences between the input and neurons.
        :param influences: A matrix containing the influence each neuron has on each
            other neuron. This is used to calculate the updates.
        :param activations: The activations each neuron has to each data point. This is
            used to calculate the BMU.
        :param diff_y: The differences between the input and context neurons.
        :return: The updates to the weights and context weights, respectively.
        """
        bmu = self._get_bmu(activations)
        influence = influences[bmu]

        # Update
        x_update = np.multiply(diff_x, influence)
        y_update = np.multiply(diff_y, influence)

        return x_update, y_update
