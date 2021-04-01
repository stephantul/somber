"""The PLSOM."""
import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from somber.som import BaseSom
from somber.components.initializers import range_initialization
from somber.components.utilities import Scaler


logger = logging.getLogger(__name__)


class PLSom(BaseSom):

    # Static property names
    param_names = {"map_dimensions", "weights", "data_dimensionality", "params"}

    def __init__(
        self,
        map_dimensions: Tuple[int],
        data_dimensionality: Optional[int] = None,
        beta: Optional[float] = None,
        initializer: Callable = range_initialization,
        scaler: Optional[Scaler] = None,
    ) -> None:
        """
        An implementation of the PLSom.

        The ParameterLess Som is a SOM which does not rely on time-induced
        plasticity adaptation. Instead, the plasticity of the SOM is adapted
        in an online fashion by continuously monitoring the error of each presented
        item.

        In general, the PLSom is less prone to catastrophic interference, or
        "forgetting" than the original SOM. Simultaneously, it is also more suited
        to re-adapting to changes in distribution. This is because the SOM loses
        its plasticity according to an exponentially decreasing learning rate and
        neighborhood size.

        :param map_dimensions: A tuple describing the map size. For example, (10, 10)
            will create a 10 * 10 map with 100 neurons, while (10, 10, 10) creates a
            10 * 10 * 10 map with 1000 neurons.
        :param data_dimensionality: The dimensionality of the input data.
        :param initializer: A function which takes in the input data and weight matrix
            and returns an initialized weight matrix. The initializers are defined in
            somber.components.initializers. Can be set to None.
        :param scaler: An initialized instance of Scaler() which is used to scale the
            data to have mean 0 and stdev 1.
        """
        super().__init__(
            map_dimensions,
            data_dimensionality=data_dimensionality,
            argfunc="argmin",
            valfunc="min",
            params={"r": {"value": 0, "factor": 1, "orig": 0}},
            initializer=initializer,
            scaler=scaler,
        )
        self.beta = beta if beta else 2

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

        This function shuffles the data internally,
        as this improves performance.

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

        # Initialize the previous activation
        prev = self._init_prev(X_)
        prev = self.distance_function(X_[0], self.weights)[0]
        influences = self._update_params(prev)

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

            # if idx > 0 and idx % update_step == 0:
            influences = self._update_params(prev)
            prev = self._propagate(x, influences, prev_activation=prev)
            if progressbar is not None:
                progressbar.update(batch_size)

    def _update_params(self, constants: np.ndarray) -> np.ndarray:
        """Update the params."""
        constants = np.max(np.min(constants, 1))
        self.params["r"]["value"] = max([self.params["r"]["value"], constants])
        epsilon = constants / self.params["r"]["value"]
        influence = self._calculate_influence(epsilon)
        # Account for learning rate
        return influence * epsilon

    def _calculate_influence(self, epsilon: float) -> np.ndarray:
        """
        Pre-calculate the influence for a given value of epsilon.

        The neighborhood has size num_neurons * num_neurons, so for a
        30 * 30 map, the neighborhood will be size (900, 900).

        :param epsilon: The neighborhood value.
        :param neighborhood: The influence from each neuron to each other neuron.
        """
        n = (self.beta - 1) * np.log(1 + epsilon * (np.e - 1)) + 1
        grid = np.exp((-self.distance_grid) / n ** 2)
        return grid.reshape(self.num_neurons, self.num_neurons)
