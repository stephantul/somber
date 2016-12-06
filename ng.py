import numpy as np
import time
import logging

from som import Som, expo, static


class NG(Som):

    def __init__(self, width, height, dim, learning_rates, lrfunc=expo, nbfunc=expo):

        super().__init__(width, height, dim, learning_rates, lrfunc, nbfunc)

    def _batch(self, batch, cache, map_radius, learning_rate):

        bmus, differences = self._get_bmus(batch)

        mask = np.arange(self.map_dim) / map_radius
        mask = np.exp(-mask / mask) * learning_rate

        influences = self._calculate_influence(mask, bmus=bmus)

        update = self._update(differences, influences).mean(axis=0)

        return update, differences

    def _get_bmus(self, x):

        differences = self._pseudo_distance(x, self.weights)
        distances = np.sqrt(np.sum(np.square(differences), axis=2))
        return np.argsort(distances, axis=1), differences

    @staticmethod
    def _calculate_influence(mask, **kwargs):

        bmus_sorted = kwargs['bmus']
        influences = np.ones_like(bmus_sorted)

        for idx, bmus in enumerate(bmus_sorted):

            influences[idx][bmus] *= mask

        return influences

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    '''colors = np.array(
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

    colors = np.array(colors)'''

    colors = []

    for x in range(10):
        for y in range(10):
            for z in range(10):
                colors.append((x/10, y/10, z/10))

    colors = np.array(colors, dtype=float)

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

    s = Som(20, 20, 3, [1.0])
    start = time.time()
    bmus = s.train(colors, num_epochs=20)

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))