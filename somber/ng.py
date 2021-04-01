"""Neural gas."""
import json
from typing import Callable, Optional

import numpy as np

from somber.base import Base, _T
from somber.components.initializers import range_initialization
from somber.components.utilities import Scaler


class Ng(Base):
    def __init__(
        self,
        num_neurons: int,
        learning_rate: float,
        influence: Optional[float] = None,
        data_dimensionality: Optional[int] = None,
        initializer: Callable = range_initialization,
        scaler: Scaler = None,
        lr_lambda: float = 2.5,
        infl_lambda: float = 2.5,
    ) -> None:
        """
        Organize your gas.

        :param num_neurons: The number of neurons in the neural gas.
        :param learning_rate: The starting learning rate.
        :param influence: The starting influence. Sane value is sqrt(num_neurons).
        :param data_dimensionality: The dimensionality of your input data.
        :param initializer: A function which takes in the input data and weight matrix
            and returns an initialized weight matrix. The initializers are defined in
            somber.components.initializers. Can be set to None.
        :param scaler: An initialized instance of Scaler() which is used to scale the
            data to have mean 0 and stdev 1.
        :param lr_lambda: Controls the steepness of the exponential function that
            decreases the learning rate.
        :param nb_lambda: Controls the steepness of the exponential function that
            decreases the neighborhood.
        """
        params = {
            "infl": {
                "value": influence,
                "factor": infl_lambda,
                "orig": np.sqrt(num_neurons),
            },
            "lr": {"value": learning_rate, "factor": lr_lambda, "orig": learning_rate},
        }

        super().__init__(
            num_neurons,
            data_dimensionality,
            params,
            "argmin",
            "min",
            initializer,
            scaler,
        )

    def _get_bmu(self, activations: np.ndarray) -> np.ndarray:
        """Get indices of bmus, sorted by their distance from input."""
        # If the neural gas is a recursive neural gas, we need reverse argsort.
        if self.argfunc == "argmax":
            activations = -activations
        sort = np.argsort(activations, 1)
        return sort.argsort()

    def _calculate_influence(self, influence_lambda: float) -> np.ndarray:
        """Calculate the ranking influence."""
        return np.exp(-np.arange(self.num_neurons) / influence_lambda)[:, None]

    @classmethod
    def load(cls, path: str) -> _T:
        """
        Load a Neural Gas from a JSON file saved with this package.

        :param path: The path to the JSON file.
        :return: A neural gas.
        """
        data = json.load(open(path))

        weights = data["weights"]
        weights = np.asarray(weights, dtype=np.float64)

        s = cls(
            data["num_neurons"],
            data["data_dimensionality"],
            data["params"]["lr"]["orig"],
            influence=data["params"]["infl"]["orig"],
            lr_lambda=data["params"]["lr"]["factor"],
            infl_lambda=data["params"]["infl"]["factor"],
        )

        s.weights = weights
        s.trained = True

        return s
