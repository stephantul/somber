"""Neural gas."""
import numpy as np
import json
from .base import Base
from .components.utilities import Scaler
from .components.initializers import range_initialization


class Ng(Base):
    """
    Neural gas.

    parameters
    ==========
    num_neurons : int
        The number of neurons in the neural gas.
    data_dimensionality : int
        The dimensionality of your input data.
    learning_rate : float
        The starting learning rate.
    influence : float
        The starting influence. Sane value is sqrt(num_neurons).
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

    """

    def __init__(self,
                 num_neurons,
                 data_dimensionality,
                 learning_rate,
                 influence,
                 initializer=range_initialization,
                 scaler=Scaler(),
                 lr_lambda=2.5,
                 infl_lambda=2.5):
        """Organize your gas."""
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

    def _get_bmu(self, activations):
        """Get indices of bmus, sorted by their distance from input."""
        # If the neural gas is a recursive neural gas, we need reverse argsort.
        if self.argfunc == 'argmax':
            activations = -activations
        sort = np.argsort(activations, 1)
        return sort.argsort()

    def _calculate_influence(self, influence_lambda):
        """Calculate the ranking influence."""
        return np.exp(-np.arange(self.num_neurons) / influence_lambda)[:, None]

    @classmethod
    def load(cls, path):
        """
        Load a Neural Gas from a JSON file saved with this package.

        parameters
        ==========
        path : str
            The path to the JSON file.

        returns
        =======
        s : cls
            A neural gas.

        """
        data = json.load(open(path))

        weights = data['weights']
        weights = np.asarray(weights, dtype=np.float32)

        s = cls(data['num_neurons'],
                data['data_dimensionality'],
                data['params']['lr']['orig'],
                influence=data['params']['infl']['orig'],
                lr_lambda=data['params']['lr']['factor'],
                infl_lambda=data['params']['infl']['factor'])

        s.weights = weights
        s.trained = True

        return s
