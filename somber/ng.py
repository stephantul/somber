"""Neural gas."""
import numpy as np
import cupy as cp
import json
from .base import Base
from .components.utilities import Scaler
from .components.initializers import range_initialization


class Ng(Base):

    def __init__(self,
                 num_neurons,
                 data_dimensionality,
                 learning_rate,
                 influence,
                 initializer=range_initialization,
                 scaler=Scaler(),
                 lr_lambda=2.5,
                 infl_lambda=2.5):

        params = {'infl': {'value': influence, 'factor': infl_lambda},
                  'lr': {'value': learning_rate, 'factor': lr_lambda}}

        super().__init__(num_neurons,
                         data_dimensionality,
                         params,
                         'argmin',
                         'min',
                         initializer,
                         scaler)

    def _get_bmu(self, activations):
        """Get indices of bmus, sorted by their distance from input."""
        xp = cp.get_array_module(activations)
        if self.argfunc == 'argmax':
            activations = -activations
        return xp.argsort(activations, 1)

    def _calculate_influence(self, influence_lambda):
        """Calculate the ranking influence."""
        return np.exp(-np.arange(self.num_neurons) / influence_lambda)[:, None]

    @classmethod
    def load(cls, path, array_type=np):
        """
        Load a Neural Gas from a JSON file saved with this package.

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
            A neural gas.

        """
        data = json.load(open(path))

        weights = data['weights']
        weights = array_type.asarray(weights, dtype=array_type.float32)

        s = cls(data['num_neurons'],
                data['data_dimensionality'],
                data['params']['lr']['value'],
                influence=data['params']['infl']['value'],
                lr_lambda=data['params']['lr']['factor'],
                infl_lambda=data['params']['infl']['factor'])

        s.weights = weights
        s.trained = True

        return s
