"""The PLSOM."""
import numpy as np
import logging

from .som import Som
from .components.initializers import range_initialization
from tqdm import tqdm


logger = logging.getLogger(__name__)


class PLSom(Som):
    """
    An implementation of the PLSom.

    The ParameterLess Som is a SOM which does not rely on time-induced
    plasticity adaptation. Instead, the plasticity of the SOM is adapted
    in an online fashion by continuously monitoring the error of each presented
    item.

    In general, the PLSom is both less prone to catastrophic interference, or
    "forgetting" than the original SOM. Simultaneously, it is also more suited
    to re-adapting to changes in distribution. This is because the SOM loses
    its plasticity according to an exponentially decreasing learning rate and
    neighborhood size.

    Parameters
    ----------
    map_dimensions : tuple
        A tuple describing the map size. For example, (10, 10) will create
        a 10 * 10 map with 100 neurons, while a (10, 10, 10) map with 1000
        neurons creates a 10 * 10 * 10 map with 1000 neurons.
    data_dimensionality : int, default None
        The dimensionality of the input data.
    initializer : function, optional, default range_initialization
        A function which takes in the input data and weight matrix and Returns
        an initialized weight matrix. The initializers are defined in
        somber.components.initializers. Can be set to None.
    scaler : initialized Scaler instance, optional default None
        An initialized instance of Scaler() which is used to scale the data
        to have mean 0 and stdev 1.

    Attributes
    ----------
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
                 data_dimensionality=None,
                 initializer=range_initialization,
                 scaler=None):
        """Organize your maps parameterlessly."""
        super().__init__(map_dimensions,
                         0,
                         initializer=initializer,
                         scaler=scaler)

        self.params = {'r': {'value': 1,
                             'factor': 1,
                             'orig': 1}}

        # Initialize the distance grid: only needs to be done once.
        self.distance_grid = self._initialize_distance_grid()

    def _epoch(self,
               X,
               epoch_idx,
               batch_size,
               updates_epoch,
               constants,
               show_progressbar):
        """
        Run a single epoch.

        This function shuffles the data internally,
        as this improves performance.

        Parameters
        ----------
        X : numpy array
            The training data.
        epoch_idx : int
            The current epoch
        batch_size : int
            The batch size
        updates_epoch : int
            The number of updates to perform per epoch
        constants : dict
            A dictionary containing the constants with which to update the
            parameters in self.parameters.
        show_progressbar : bool
            Whether to show a progressbar during training.

        """
        # Create batches
        X_ = self._create_batches(X, batch_size)
        X_len = np.prod(X.shape[:-1])

        # Initialize the previous activation
        prev = self._init_prev(X_)
        dist = self.distance_function(X_[0][None, :], self.weights)[0]
        influences = self._update_params(dist)

        # Iterate over the training data
        for idx, x in enumerate(tqdm(X_, disable=not show_progressbar)):

            # Our batches are padded, so we need to
            # make sure we know when we hit the padding
            # so we don't inadvertently learn zeroes.
            diff = X_len - (idx * batch_size)
            if diff and diff < batch_size:
                x = x[:diff]
                # Prev_activation may be None
                if prev is not None:
                    prev = prev[:diff]

            if idx > 0 and idx % updates_epoch == 0:
                influences = self._update_params(prev)
                logger.info(self.params)

            prev = self._propagate(x,
                                   influences,
                                   prev_activation=prev)

    def _update_params(self, constants):
        """Update the params."""
        constants = np.max(np.min(constants, 1))
        self.params['r']['value'] = max([self.params['r']['value'],
                                         constants])

        influence = self._calculate_influence(constants /
                                              self.params['r']['value'])

        return influence

    def _calculate_influence(self, neighborhood):
        """
        Pre-calculate the influence for a given value of sigma.

        The neighborhood has size num_neurons * num_neurons, so for a
        30 * 30 map, the neighborhood will be size (900, 900).

        Parameters
        ----------
        neighborhood : float
            The neighborhood value.

        Returns
        -------
        neighborhood : numpy array
            The influence from each neuron to each other neuron.

        """
        grid = np.exp(-(self.distance_grid) / (neighborhood ** 2))
        grid *= neighborhood
        return grid.reshape(self.num_neurons, self.num_neurons)[:, :, None]
