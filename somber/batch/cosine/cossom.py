import numpy as np

from ..som import Som
from somber.utils import np_max


class CosSom(Som):

    def __init__(self,
                 map_dim,
                 data_dim,
                 learning_rate):

        super().__init__(map_dim, data_dim, learning_rate, min_max=np_max)

    def _create_batches(self, X, batch_size):

        self.progressbar_interval = 1
        self.progressbar_mult = batch_size

        max_x = int(np.ceil(X.shape[0] / batch_size))
        X = np.resize(X, (max_x, batch_size, X.shape[1]))

        return X

    def distance_function(self, x, weights):

        w = np.square(weights / np.linalg.norm(weights, axis=1)[:, None])
        x = np.square(x / np.linalg.norm(x, axis=1)[:, None])

        x = np.nan_to_num(x)

        return x.dot(w.T), self._distance_difference(x, weights)

