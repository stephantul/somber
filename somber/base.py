"""Base class for SOM and Neural Gas."""
import json
import logging
import time
import types
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
from tqdm import tqdm

from somber.components.utilities import shuffle, Scaler
from somber.components.initializers import range_initialization
from somber.distance import euclidean


# For class methods
_T = TypeVar("_T")

logger = logging.getLogger(__name__)


class Base(object):
    """
    This is a base class for the Neural gas and SOM.

    """

    # Static property names
    param_names = {
        "num_neurons",
        "weights",
        "data_dimensionality",
        "params",
        "valfunc",
        "argfunc",
    }

    def __init__(
        self,
        num_neurons: Tuple[int],
        data_dimensionality: Optional[int],
        params: Dict[str, float],
        argfunc: str = "argmin",
        valfunc: str = "min",
        initializer: Callable = range_initialization,
        scaler: Optional[Scaler] = None,
    ) -> None:
        """
        Organize nothing.

        :param num_neurons: The number of neurons to create.
        :param data_dimensionality: The dimensionality of the input data. Set to None
            to infer it from the data.
        :param params: A dictionary describing the parameters which need to be reduced
            over time. Each parameter is denoted by two fields: "value" and
            "factor", which denote the current value, and the constant factor
            with which the value is multiplied in each update step.
        :param argfunc: The name of the function which is used for calculating the index of
            the BMU.
        :param valfunc: The name of the function which is used for calculating the value of the
            BMU.
        :param initializer: A function which takes in the input data and weight matrix and returns
            an initialized weight matrix. The initializers are defined in
            somber.components.initializers. Can be set to None.
        :param scaler: An initialized instance of Scaler which is used to scale the data
            to have mean 0 and stdev 1. If this is set to None, the SOM will
            create a scaler.
        """
        self.num_neurons = np.int(num_neurons)
        self.data_dimensionality = data_dimensionality
        if self.data_dimensionality:
            self.weights = np.zeros((num_neurons, data_dimensionality))
        else:
            self.weights = None
        self.argfunc = argfunc
        self.valfunc = valfunc
        self.trained = False
        if scaler is None:
            self.scaler = Scaler()
        self.initializer = initializer
        self.params = params
        self.scaler = scaler

    def fit(
        self,
        X: np.ndarray,
        num_epochs: int = 10,
        updates_epoch: Optional[int] = None,
        stop_param_updates: Optional[Dict[str, int]] = None,
        batch_size: int = 32,
        show_progressbar: bool = False,
        refit: bool = True,
    ) -> _T:
        """
        Fit the learner to some data.

        :param X: The input data.
        :param num_epochs: The number of epochs to train for.
        :param updates_epoch: The number of updates to perform on the learning rate and
            neighborhood per epoch. 10 suffices for most problems.
        :param stop_param_updates: The epoch at which to stop updating each param.
            This means that the specified parameter will be reduced to 0 at the
            specified epoch. If this is None, all params become 0 at the end.
        :param batch_size: The batch size to use. Warning: batching can change your
            performance dramatically, depending on the task.
        :param show_progressbar: Whether to show a progressbar during training.
        """
        if self.data_dimensionality is None:
            self.data_dimensionality = X.shape[-1]
            self.weights = np.zeros((self.num_neurons, self.data_dimensionality))
        X = self._check_input(X)
        if not self.trained or refit:
            X = self._init_weights(X)
        else:
            if self.scaler is not None:
                self.weights = self.scaler.transform(self.weights)

        if updates_epoch is None:
            X_len = X.shape[0]
            updates_epoch = np.min([50, X_len // batch_size])

        stop_param_updates = stop_param_updates or {}

        constants = self._pre_train(stop_param_updates, num_epochs, updates_epoch)
        start = time.time()

        progressbar = tqdm(total=len(X) * num_epochs) if show_progressbar else None

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1} of {num_epochs}")
            self._epoch(X, batch_size, updates_epoch, constants, progressbar)

        if progressbar is not None:
            progressbar.close()
        self.trained = True
        if self.scaler is not None:
            self.weights = self.scaler.inverse_transform(self.weights)
        logger.info(f"Total train time: {time.time() - start}")

        return self

    def _init_weights(self, X: np.ndarray) -> np.ndarray:
        """Set the weights and normalize data before starting training."""
        X = np.asarray(X, dtype=np.float64)

        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        if self.initializer is not None:
            self.weights = self.initializer(X, self.num_neurons)

        for v in self.params.values():
            v["value"] = v["orig"]

        return X

    def _pre_train(
        self, stop_param_updates: Dict[str, int], num_epochs: int, updates_epoch: int
    ) -> Dict[str, float]:
        """Set parameters and constants before training."""
        # Calculate the total number of updates given early stopping.
        updates = {
            k: stop_param_updates.get(k, num_epochs) * updates_epoch
            for k, v in self.params.items()
        }

        # Calculate the value of a single step given the number of allowed
        # updates.
        single_steps = {
            k: np.exp(-((1.0 - (1.0 / v))) * self.params[k]["factor"])
            for k, v in updates.items()
        }

        # Calculate the factor given the true factor and the value of a
        # single step.
        constants = {
            k: np.exp(-self.params[k]["factor"]) / v for k, v in single_steps.items()
        }

        return constants

    def fit_predict(
        self,
        X: np.ndarray,
        num_epochs: int = 10,
        updates_epoch: int = 10,
        stop_param_updates: Optional[Dict[str, int]] = None,
        batch_size: int = 1,
        show_progressbar: bool = False,
    ) -> np.ndarray:
        """First fit, then predict."""
        self.fit(
            X,
            num_epochs,
            updates_epoch,
            stop_param_updates,
            batch_size,
            show_progressbar,
        )

        return self.predict(X, batch_size=batch_size)

    def fit_transform(
        self,
        X: np.ndarray,
        num_epochs: int = 10,
        updates_epoch: int = 10,
        stop_param_updates: Optional[Dict[str, int]] = None,
        batch_size: int = 1,
        show_progressbar: bool = False,
        show_epoch: bool = False,
    ) -> np.ndarray:
        """First fit, then transform."""
        self.fit(
            X,
            num_epochs,
            updates_epoch,
            stop_param_updates,
            batch_size,
            show_progressbar,
            show_epoch,
        )

        return self.transform(X, batch_size=batch_size)

    def _epoch(
        self,
        X: np.ndarray,
        batch_size: int,
        updates_epoch: int,
        constants: Dict[str, float],
        progressbar: tqdm,
    ) -> None:
        """
        Run a single epoch.

        :param X: The training data.
        :param batch_size: The batch size
        :param updates_epoch: The number of updates to perform per epoch
        :param constants: A dictionary containing the constants with which to update the
            parameters in self.parameters.
        :param progressbar: The progressbar instance to show and update during training
        """
        # Create batches
        X_ = self._create_batches(X, batch_size)
        X_len = np.prod(X.shape[:-1])

        update_step = np.ceil(X_.shape[0] / updates_epoch)

        # Initialize the previous activation
        prev = self._init_prev(X_)
        influences = self._update_params(constants)

        # Iterate over the training data
        for idx, x in enumerate(X_):

            # Our batches are padded, so we need to
            # make sure we know when we hit the padding
            # so we don't inadvertently learn zeroes.
            diff = X_len - (idx * batch_size)
            if diff and diff < batch_size:
                x = x[:diff]
                # Prev_activation may be None
                if prev is not None:
                    prev = prev[:diff]

            # If we hit an update step, perform an update.
            if idx % update_step == 0:
                influences = self._update_params(constants)
                logger.info(self.params)

            prev = self._propagate(x, influences, prev_activation=prev)
            if progressbar is not None:
                progressbar.update(batch_size)

    def _update_params(self, constants: Dict[str, float]) -> np.ndarray:
        """Update params and return new influence."""
        for k, v in constants.items():
            self.params[k]["value"] *= v

        influence = self._calculate_influence(self.params["infl"]["value"])
        return influence * self.params["lr"]["value"]

    def _init_prev(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Initialize recurrent SOMs."""
        return None

    def _get_bmu(self, activations: np.ndarray) -> np.ndarray:
        """Get bmu based on activations."""
        return activations.__getattribute__(self.argfunc)(1)

    def _create_batches(
        self, X: np.ndarray, batch_size: int, shuffle_data: bool = True
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
        X = np.resize(X, (max_x, batch_size, X.shape[-1]))

        return X

    def _propagate(self, x: np.ndarray, influences: np.ndarray, **kwargs) -> np.ndarray:
        """Propagate a single batch of examples through the network."""
        activation, difference_x = self.forward(x)
        update = self.backward(difference_x, influences, activation)
        self.weights += update.mean(0)
        return activation

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Get the best matching neurons, and the difference between inputs.

        Note: it might seem like this function can be replaced by a call to
        distance function. This is only true for the regular SOM. other
        SOMs, like the recurrent SOM need more complicated forward pass
        functions.

        :param x: The input vectors.
        :return: A tuple containing the activations and differences between neurons
            and input, respectively.
        """
        return self.distance_function(x, self.weights)

    def backward(
        self,
        diff_x: np.ndarray,
        influences: np.ndarray,
        activations: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Backward pass through the network, including update.

        :param diff_x: A matrix containing the differences between the input and
            neurons.
        :param influences: A matrix containing the influence each neuron has on each
            other neuron. This is used to calculate the updates.
        :param activations: The activations each neuron has to each data point. This is
            used to calculate the BMU.
        :return: The updates to be applied.
        """
        bmu = self._get_bmu(activations)
        influence = influences[bmu]
        update = influence[:, :, None] * diff_x
        return update

    def distance_function(
        self, x: np.ndarray, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate euclidean distance between a batch of input data and weights.

        :param x: The input data.
        :param weights: The weights
        :return: A tuple of distances from each input to each weight, and the
            feature-wise differences between each input and each weight
        """
        distances, differences = euclidean(x, weights)
        # For clarity
        return distances, differences

    def _check_input(self, X: np.ndarray) -> np.ndarray:
        """
        Check the input for validity.

        Ensures that the input data, X, is a 2-dimensional matrix, and that
        the second dimension of this matrix has the same dimensionality as
        the weight matrix.
        """
        if np.ndim(X) == 1:
            X = np.reshape(X, (1, -1))

        if X.ndim != 2:
            raise ValueError(f"Your data is not a 2D matrix. Actual size: {X.shape}")

        if X.shape[1] != self.data_dimensionality:
            raise ValueError(
                f"Your data size != weight dim: {X.shape[1]}, "
                f"expected {self.data_dimensionality}"
            )
        return X

    def transform(
        self, X: np.ndarray, batch_size: int = 100, show_progressbar: bool = False
    ) -> np.ndarray:
        """
        Transform input to a distance matrix by measuring the L2 distance.

        :param X: The input data.
        :param batch_size: The batch size to use in transformation. This may affect the
            transformation in stateful, i.e. sequential SOMs.
        :param show_progressbar: Whether to show a progressbar during transformation.
        :return: An array of activations for each input item.
        """
        X = self._check_input(X)

        batched = self._create_batches(X, batch_size, shuffle_data=False)

        activations = []
        prev = self._init_prev(batched)

        for x in tqdm(batched, disable=not show_progressbar):
            prev = self.forward(x, prev_activation=prev)[0]
            activations.extend(prev)

        activations = np.asarray(activations, dtype=np.float64)
        activations = activations[: X.shape[0]]
        return activations.reshape(X.shape[0], self.num_neurons)

    def predict(
        self, X: np.ndarray, batch_size: int = 32, show_progressbar: bool = False
    ) -> np.ndarray:
        """
        Predict the BMU for each input data.

        :param X: The input data.
        :param batch_size: The batch size to use in prediction. This may affect prediction
            in stateful, i.e. sequential SOMs.
        :param show_progressbar : Whether to show a progressbar during prediction.
        :return: An array containing the BMU for each input data point.
        """
        dist = self.transform(X, batch_size, show_progressbar)
        res = dist.__getattribute__(self.argfunc)(1)

        return res

    def quantization_error(self, X, batch_size=1):
        """
        Calculate the quantization error.

        Find the the minimum euclidean distance between the units and
        some input.

        :param X: The input data.
        :param batch_size: The batch size to use for processing.

        :return: The error for each data point.
        """
        dist = self.transform(X, batch_size)
        res = dist.__getattribute__(self.valfunc)(1)

        return res

    def receptive_field(
        self, X: np.ndarray, identities: List, max_len=10, threshold=0.9, batch_size=1
    ) -> Dict[int, List[List[str]]]:
        """
        Calculate the receptive field of the SOM on some data.

        The receptive field is the common ending of all sequences which
        lead to the activation of a given BMU. If a SOM is well-tuned to
        specific sequences, it will have longer receptive fields, and therefore
        gives a better description of the dynamics of a given system.

        :param X: Input data.
        :param identities: A list of symbolic identities associated with each input.
            We enpect this list to be as long as the input data.
        :param max_len: The maximum length to attempt to find. Raising this increases
            memory use.
        :param threshold: The threshold at which we consider a receptive field valid.
            If at least this proportion of the sequences of a neuron have the same
            suffix, that suffix is counted as acquired by the SOM.
        :param batch_size: The batch size to use in prediction

        :return: A dictionary mapping from the neuron id to the found sequences for
            that neuron. The sequences are represented as lists of symbols of identities.
        """
        receptive_fields = defaultdict(list)

        predictions = self.predict(X)

        if len(predictions) != len(identities):
            raise ValueError(
                "X and identities are not the same length: "
                f"{len(predictions)} and {len(identities)}"
            )

        for idx, p in enumerate(predictions.tolist()):
            receptive_fields[p].append(identities[idx + 1 - max_len : idx + 1])

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

    @classmethod
    def load(cls, path: str) -> _T:
        """
        Load a SOM from a JSON file saved with this package.

        :param path: The path to the JSON file.
        :return: A som of the specified class.
        """
        data = json.load(open(path))

        weights = data["weights"]
        weights = np.asarray(weights, dtype=np.float64)

        s = cls(
            data["num_neurons"],
            data["data_dimensionality"],
            data["params"]["lr"]["orig"],
            neighborhood=data["params"]["infl"]["orig"],
            valfunc=data["valfunc"],
            argfunc=data["argfunc"],
            lr_lambda=data["params"]["lr"]["factor"],
            nb_lambda=data["params"]["nb"]["factor"],
        )

        s.weights = weights
        s.trained = True

        return s

    def save(self, path: str) -> None:
        """Save a SOM to a JSON file."""
        to_save = {}
        for x in self.param_names:
            attr = self.__getattribute__(x)
            if type(attr) == np.ndarray:
                attr = [[float(x) for x in row] for row in attr]
            elif isinstance(attr, types.FunctionType):
                attr = attr.__name__
            to_save[x] = attr

        json.dump(to_save, open(path, "w"))
