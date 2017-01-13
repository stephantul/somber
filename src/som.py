import logging
import time
import numpy as np

from utils import MultiPlexer, expo, linear
from base import Base


logger = logging.getLogger(__name__)


class Som(Base):

    def __init__(self, map_dim, dim, learning_rate, lrfunc=expo, nbfunc=expo, sigma=None):

        super().__init__(map_dim=map_dim,
                         dim=dim,
                         learning_rate=learning_rate,
                         lrfunc=lrfunc,
                         nbfunc=nbfunc,
                         sigma=sigma,
                         apply_influences=self._apply_influences,
                         calculate_distance_grid=self._init_distance_grid,
                         calculate_influence=self._calc_influence)

    def _apply_influences(self, distances, influences):

        bmu = np.argmin(distances)
        return influences[bmu], bmu

    def _calc_influence(self, sigma):
        """


        :param sigma:
        :return:
        """

        neighborhood = np.exp(-1.0 * self.distance_grid / (2.0 * sigma ** 2)).reshape(self.map_dim, self.map_dim)
        return np.asarray([neighborhood] * self.data_dim).transpose((1, 2, 0))

    def _init_distance_grid(self):
        """


        :return:
        """

        distance_matrix = np.zeros((self.map_dim, self.map_dim))

        for i in range(self.map_dim):

            distance_matrix[i] = self._grid_dist(i).reshape(1, self.map_dim)

        return distance_matrix

    def _grid_dist(self, index):
        """


        :param index:
        :return:
        """

        width, height = self.map_dimensions

        column = int(index % width)
        row = index // width

        r = np.arange(height)[:, np.newaxis]
        c = np.arange(width)
        distance = (r-row)**2 + (c-column)**2

        return distance.ravel()

    def map_weights(self):
        """
        Retrieves the grid as a list of lists of weights. For easy visualization.

        :return: A three-dimensional Numpy array of values (width, height, data_dim)
        """

        width, height = self.map_dimensions

        return self.weights.reshape((width, height, self.data_dim)).transpose(1, 0, 2)

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

    '''colors = []

    for x in range(10):
        for y in range(10):
            for z in range(10):
                colors.append((x/10, y/10, z/10))

    colors_ = np.array(colors, dtype=float)'''

    colors = MultiPlexer(colors_, 1000)

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

    s = Som((20, 20), 3, 0.3, sigma=10, nbfunc=linear)
    start = time.time()
    bmus = s.train(colors, total_epochs=100, rough_epochs=0.5)

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))

    '''from visualization.umatrix import UMatrixView

    view = UMatrixView(500, 500, 'dom')
    view.create(s.weights, colors, s.width, s.height, bmus[-1])
    view.save("junk_viz/_{0}.svg".format(0))

    print("Made {0}".format(0))'''