import numpy as np
import time
import logging
import cProfile

from progressbar import progressbar
from som import Som


logger = logging.getLogger(__name__)


class Rsom(Som):

    def __init__(self, width, height, dim, learning_rate, temp_weight):

        self.temp_weight = temp_weight
        super().__init__(width, height, dim, learning_rate)

    def epoch_step(self, X, map_radius, learning_rate, batch_size):
        """
        A single example.

        :param X: a numpy array of examples
        :param map_radius: The radius at the current epoch, given the learning rate and map size
        :param learning_rate: The learning rate.
        :param batch_size: The batch size
        :param return_activations: Whether to return activations instead of bmus. For processing in
         another network.
        :return: The best matching unit
        """

        # Calc once per epoch
        self.grid, self.grid_distances = self._distance_grid(map_radius)

        learning_rate = learning_rate[0]

        # One radius per epoch
        map_radius_squared = (2 * map_radius) ** 2

        # One cache per epoch
        cache = {}

        # One accumulator per epoch
        # bmus = np.zeros((X.shape[0], X.shape[1]), dtype=np.int)
        all_distances = np.zeros((self.map_dim,))

        # Make a batch generator.
        accumulator = np.zeros_like(self.weights)
        num_updates = 0

        num_batches = np.ceil(len(X) / batch_size).astype(int)

        for index in progressbar(range(num_batches), idx_interval=1, mult=batch_size):

            # Select the current batch.
            current = X[index * batch_size: (index+1) * batch_size]

            # Initial previous activation
            prev_activations = np.zeros((current.shape[0], self.map_dim, self.data_dim))

            for idx in range(current.shape[1]):

                column = current[:, idx, :]

                influences = []

                # Get the indices of the Best Matching Unit, given the data.
                bmu_theta, distances, prev_activations = self._get_bmus(column, previous=prev_activations)

                all_distances += distances.mean(axis=0)

                for bmu in bmu_theta:

                    try:
                        influence = cache[bmu]
                    except KeyError:

                        x_, y_ = self._index_dict[bmu]
                        influence = self._calculate_influence(map_radius_squared, center_x=x_, center_y=y_)
                        influence = np.tile(influence, (self.data_dim, 1)).T * learning_rate
                        cache[bmu] = influence

                    influences.append(influence)

                influences = np.array(influences)

                update = self._update(prev_activations, influences)

                accumulator += update.mean(axis=0)
                num_updates += 1

        self.weights += (accumulator / num_updates)

        return np.array(all_distances / num_updates)

    def _get_bmus(self, x, **kwargs):
        """
        Gets the best matching units, based on euclidean distance.

        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        differences = self._pseudo_distance(x, self.weights)

        differences = ((1 - self.temp_weight) * kwargs['previous']) + (self.temp_weight * differences)
        distances = np.sqrt(np.sum(np.square(differences), axis=2))

        return np.argmin(distances, axis=1), distances, differences

    def predict(self, X):
        """
        Predicts node identity for input data.
        Similar to a clustering procedure.

        :param x: The input data.
        :return: A list of indices
        """

        # Start with a clean buffer.

        prev_activations = np.zeros((len(X), self.map_dim, self.data_dim))

        all_bmus = []

        for idx in range(X.shape[1]):

            column = X[:, idx, :]
            bmus, distances, prev_activations = self._get_bmus(column, previous=prev_activations)

            all_bmus.append(bmus)

        return np.array(all_bmus).T

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    colors = np.array(
         [[1., 0., 1.],
          [0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])

    data = np.tile(colors, (100, 1, 1))

    colorpicker = np.arange(len(colors))

    data = np.random.choice(colorpicker, size=(1000, 15))
    data = colors[data]

    s = Rsom(30, 30, 3, [1.0], 1.0)
    start = time.time()
    bmus = s.train(data, num_epochs=100, batch_size=100)

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))
