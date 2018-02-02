"""Som with lateral and afferent connections."""
import numpy as np
from somber.som import Som
from somber.components.initializers import range_initialization
from somber.components.utilities import Scaler


"""
W = weights of nodes (Z, N)?
X = batched input (M, N)
Waff = afferent weights (N, Z)
"""

class LisSom(Som):

    def __init__(self,
                 map_dimensions,
                 data_dimensionality,
                 learning_rate,
                 alpha=.9,
                 beta=.9,
                 pos_neighborhood=4,
                 neg_neighborhood=12,
                 neg_thresh=0.,
                 pos_thresh=10,
                 influence=None,
                 initializer=range_initialization,
                 scaler=None,
                 lr_lambda=2.5,
                 infl_lambda=2.5):

        self.alpha = alpha
        self.beta = beta
        self.pos_neighborhood = pos_neighborhood
        self.neg_neighborhood = neg_neighborhood

        super().__init__(map_dimensions,
                         data_dimensionality,
                         learning_rate,
                         influence,
                         initializer,
                         scaler,
                         lr_lambda,
                         infl_lambda)

        self.afferent = np.random.uniform(.0, .1, (self.num_neurons, data_dimensionality))
        self.lateral_exc = np.zeros((self.num_neurons, self.num_neurons))
        self.lateral_inh = np.zeros((self.num_neurons, self.num_neurons))
        for idx, x in enumerate(self.distance_grid):
            pos = np.random.uniform(size=len(np.flatnonzero(x > 0)))
            self.lateral_exc[idx][x > 0] = pos / pos.sum()
            z = x != 0
            neg = np.random.uniform(size=len(np.flatnonzero(z)))
            self.lateral_inh[idx][z] = neg / neg.sum()

        self.neg = neg_thresh
        self.pos = pos_thresh

    def _propagate(self, x, influences, **kwargs):
        """Propagate a single batch of examples through the network."""
        activation = np.zeros((self.num_neurons))
        for _ in range(15):
            activation = self.activation(x, activation)

        # Try to only update things with positive or negative links.
        z = np.outer(activation, x)
        self.afferent += z
        self.afferent /= np.linalg.norm(self.afferent, axis=1)[:, None]

        # Hebb rule, activation of each neuron with each neuron
        p = np.outer(activation, activation)

        self.lateral_inh[self.distance_grid != 0] += p[self.distance_grid != 0]
        self.lateral_inh /= self.lateral_inh.sum(axis=1)[:, None]

        self.lateral_exc[self.distance_grid > 0] += p[self.distance_grid > 0]
        self.lateral_exc /= self.lateral_exc.sum(axis=1)[:, None]

        return activation

    def step(self, x):
        """Calculate the batched sigmoid."""
        x = np.copy(x)
        scale = 1. / (self.pos - self.neg)
        x -= self.neg
        x *= scale
        return x.clip(0., 1.)

    def forward(self, X, tol=.0001):
        """Do a forward pass."""
        pass

    def activation(self, x, prev):
        """Calculate activation."""
        # Afferent
        afferent = x.dot(self.afferent.T)
        # Lateral
        pos = prev.dot(self.lateral_exc) * self.alpha
        neg = prev.dot(self.lateral_inh) * self.beta
        activation = afferent + pos - neg
        # activation = afferent
        return self.step(activation)

    def _initialize_distance_grid(self):
        """Initialize the distance grid by calls to _grid_dist."""
        p = np.array([self._grid_distance(i) for i in range(self.num_neurons)])
        res = np.zeros_like(p, dtype=np.float32)

        for idx, x in enumerate(p):
            inhibitors = np.logical_and(x >= self.pos_neighborhood, x < self.neg_neighborhood)
            exciters = x < self.pos_neighborhood
            z = -1
            res[idx, inhibitors] = z
            z = 1
            res[idx, exciters] = z

        return res.reshape(len(p), len(p))

    def _grid_distance(self, index):
        """
        Calculate the distance grid for a single index position.

        This is pre-calculated for fast neighborhood calculations
        later on (see _calc_influence).
        """
        # Take every dimension but the first in reverse
        # then reverse that list again.
        dimensions = np.cumprod(self.map_dimensions[1::][::-1])[::-1]

        coord = []
        for idx, dim in enumerate(dimensions):
            if idx != 0:
                value = (index % dimensions[idx-1]) // dim
            else:
                value = index // dim
            coord.append(value)

        coord.append(index % self.map_dimensions[-1])

        for idx, (width, row) in enumerate(zip(self.map_dimensions, coord)):
            x = np.abs(np.arange(width) - row)
            dims = self.map_dimensions[::-1]
            if idx:
                dims = dims[:-idx]
            x = np.broadcast_to(x, dims).T
            if idx == 0:
                distance = np.copy(x)
            else:
                x = x.repeat(distance.shape[0]).reshape(distance.shape).T
                distance = np.stack([x, distance]).max(0)

        return distance


if __name__ == "__main__":

    X = np.array([[0., 0., 0.],
                [0., 0., 1.],
                [0., 0., 0.5],
                [0.125, 0.529, 1.0],
                [0.33, 0.4, 0.67],
                [0.6, 0.5, 1.0],
                [0., 1., 0.],
                [1., 0., 0.],
                [0., 1., 1.],
                [1., 0., 1.],
                [1., 1., 0.],
                [1., 1., 1.],
                [.33, .33, .33],
                [.5, .5, .5],
                [.66, .66, .66]])

    l = LisSom((20, 20), 3, 1.0, neg_thresh=.2, pos_thresh=.9, scaler=None)
    from matplotlib import pyplot as plt

    l.fit(X, num_epochs=1000, show_progressbar=False)
