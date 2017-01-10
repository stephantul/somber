import numpy as np
import time
import logging

from utils import static, expo, MultiPlexer
from base import Base


class Ng(Base):

    def __init__(self, num_nodes, dim, learning_rate, sigma, lrfunc=expo, nbfunc=expo):

        super().__init__(num_nodes,
                         dim,
                         learning_rate,
                         lrfunc=lrfunc,
                         nbfunc=nbfunc,
                         sigma=sigma,
                         apply_influences=self._apply_influences,
                         calculate_distance_grid=self._init_distance_grid,
                         calculate_influence=self._calc_influence)

    def _init_distance_grid(self):

        return np.arange(self.map_dim)

    def _calc_influence(self, sigma):

        i = np.exp(-self.distance_grid / sigma)
        return np.asarray([i] * self.data_dim).transpose()

    def _apply_influences(self, distances, influences):

        s = np.argsort(distances)
        return influences[s], s[0]

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    colors_ = np.array(
         [[0., 0., 0.],
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

    colors = []

    for x in range(10):
        for y in range(10):
            for z in range(10):
                colors.append((x/10, y/10, z/10))

    colors_ = np.array(colors, dtype=float)

    colors = MultiPlexer(colors_, 100)

    # colors = np.vstack([colors, colors, colors, colors, colors, colors, colors, colors])

    '''addendum = np.arange(len(colors) * 10).reshape(len(colors) * 10, 1) / 10

    colors = np.array(colors)
    colors = np.repeat(colors, 10).reshape(colors.shape[0] * 10, colors.shape[1])

    print(colors.shape, addendum.shape)

    colors = np.hstack((colors,addendum))
    print(colors.shape)'''

    color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'greyblue', 'lilac', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'darkgrey', 'mediumgrey', 'lightgrey']

    n = Ng((400,), 3, 0.3, sigma=0.0001)
    start = time.time()
    bmus = n.train(colors, num_effective_epochs=100)

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))