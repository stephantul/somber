import numpy as np
import time
import logging
import cProfile

from progressbar import progressbar
from som import Som, expo, static


logger = logging.getLogger(__name__)


class MSom(Som):

    def __init__(self, width, height, dim, learning_rate, alpha, beta, lrfunc=expo, nbfunc=expo):

        self.alpha = alpha
        self.beta = beta
        super().__init__(width, height, dim, learning_rate, lrfunc, nbfunc)

        self.context_weights = np.zeros_like(self.weights)

    def epoch_step(self, X, map_radius, learning_rate, batch_size):
        """
        A single epoch.

        :param X: a numpy array of data
        :param map_radius: The radius at the current epoch, given the learning rate and map size
        :param learning_rate: The learning rate.
        :param batch_size: The batch size
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
        accumulator_x = np.zeros_like(self.weights)
        accumulator_y = np.zeros_like(self.context_weights)

        # Make a batch generator.
        num_updates = 0

        num_batches = np.ceil(len(X) / batch_size).astype(int)

        all_activations = []

        for index in progressbar(range(num_batches), idx_interval=1, mult=batch_size):

            # Select the current batch.
            current = X[index * batch_size: (index+1) * batch_size]

            # Weight vector of previous activations.
            # ct = (1 - beta) * W_i + beta * cI
            # both are shape of data_dim
            prev_activations = np.zeros((current.shape[0], self.data_dim))

            for idx in range(current.shape[1]):

                # Select a column, which represents the idxth example of a sequence.
                column = current[:, idx, :]

                # Reset influence.
                influences = []

                # Get the indices of the Best Matching Units, given the data.
                bmu_theta, x_activations, y_activations = self._get_bmus(column, y=prev_activations)

                p = np.arange(len(bmu_theta))
                prev_activations = (1 - self.beta) * x_activations[p, bmu_theta] + self.beta * y_activations[p, bmu_theta]

                for bmu in bmu_theta:

                    # There is only one possible influence per BMU, so caching works
                    # really well, especially for a small number of exemplars.
                    try:
                        influence = cache[bmu]
                    except KeyError:

                        x_, y_ = self._index_dict[bmu]
                        influence = self._calculate_influence(map_radius_squared, center_x=x_, center_y=y_)

                        # Multiply the influence by the learning rate for speed.
                        influence = np.tile(influence * learning_rate, (self.data_dim, 1)).T
                        cache[bmu] = influence

                    influences.append(influence)

                # Influences is same size as minibatches.
                influences = np.array(influences)

                # Minibatch update of X and Y. Returns arrays of updates,
                # one for each example.
                update_x = self._update(x_activations, influences)
                update_y = self._update(y_activations, influences)

                # Take the mean of all updates, and update
                # the accumulator (not the weights!)
                accumulator_x += update_x.mean(axis=0)
                accumulator_y += update_y.mean(axis=0)
                num_updates += 1
                # all_activations.append((x_activations, y_activations))

        # Update the weights and recursive weights only
        # once per epoch with the mean of updates.
        # works because the mean of a mean is the same as the mean of
        # the original set of observations.
        self.weights += (accumulator_x / num_updates)
        self.context_weights += (accumulator_y / num_updates)

        return all_activations

    def _get_bmus(self, x, **kwargs):
        """
        Gets the best matching units, based on euclidean distance.

        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        # Differences is the components of the weights subtracted from the weight vector.
        differences_x = self._pseudo_distance(x, self.weights)
        # Idem for previous activation.
        differences_y = self._pseudo_distance(kwargs['y'], self.context_weights)

        # Distances are squared euclidean norm of differences.
        # Since euclidean norm is sqrt(sum(square(x)))) we can leave out the sqrt
        # and avoid doing an extra square.
        # Axis 2 because we are doing minibatches.
        distances_x = np.sum(np.square(differences_x), axis=2)
        distances_y = np.sum(np.square(differences_y), axis=2)

        # BMU is based on a weigted subtraction of current and previous activation.
        distances = ((1 - self.alpha) * distances_x) + (self.alpha * distances_y)

        return np.argmin(distances, axis=1), differences_x, differences_y

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
            bmus, x_activations, y_activations = self._get_bmus(column, y=prev_activations)
            p = np.arange(len(bmus))
            prev_activations = (1 - self.beta) * x_activations[p, bmus] + self.beta * y_activations[p, bmus]

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

    s = MSom(30, 30, 3, [1.0], 0.03, 0.5)
    start = time.time()
    bmus = s.train(data, num_epochs=1000, batch_size=100)

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))
