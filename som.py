import numpy as np
import time
import logging
import cProfile

from progressbar import progressbar
from collections import defaultdict

logger = logging.getLogger(__name__)


def expo(value, current_epoch, total_epochs):

    return value * np.exp(-current_epoch / total_epochs)


def static(value, current_epoch, total_epochs):

    return value


class Som(object):

    def __init__(self, width, height, dim, learning_rates, lrfunc=expo, nbfunc=expo):

        self.scaling_factor = max(width, height) / 2
        self.scaler = 0

        if type(learning_rates) != list:
            learning_rates = [learning_rates]

        self.learning_rates = np.array(learning_rates)

        self.width = width
        self.height = height
        self.weights = np.random.normal(0, 0.1, size=(width * height, dim))
        self.grid = None
        self.grid_distances = None
        self.data_dim = dim
        self.map_dim = width * height

        self._index_dict = {idx: (idx // self.height, idx % self.height) for idx in range(self.weights.shape[0])}
        self._coord_dict = defaultdict(dict)

        self.lrfunc = lrfunc
        self.nbfunc = nbfunc

        for k, v in self._index_dict.items():

            x_, v_ = v
            self._coord_dict[x_][v_] = k

        self.trained = False

    def train(self, X, num_epochs=10, batch_size=100):
        """
        Fits the SOM to some data for a number of epochs.
        As the learning rate is decreased proportionally to the number
        of epochs, incrementally training a SOM is not feasible.

        :param X: the data on which to train.
        :param num_epochs: The number of epochs to simulate
        :return: None
        """

        # Scaler ensures that the neighborhood radius is 0 at the end of training
        # given a square map.
        self.scaler = num_epochs / np.log(self.scaling_factor)

        # Local copy of learning rate.
        learning_rate = self.learning_rates

        bmus = []

        real_start = time.time()

        for epoch in range(num_epochs):

            print("\nEPOCH: {0}/{1}".format(epoch+1, num_epochs))
            start = time.time()

            map_radius = self.nbfunc(self.scaling_factor, epoch, num_epochs)
            bmu = self.epoch_step(X, map_radius, learning_rate, batch_size=batch_size)

            bmus.append(bmu)
            learning_rate = self.lrfunc(self.learning_rates, epoch, num_epochs)

            print("\nEPOCH TOOK {0:.2f} SECONDS.".format(time.time() - start))
            print("TOTAL: {0:.2f} SECONDS.".format(time.time() - real_start))

        self.trained = True

        return bmus

    def epoch_step(self, X, map_radius, learning_rate, batch_size):
        """
        A single example.

        :param X: a numpy array of examples
        :param map_radius: The radius at the current epoch, given the learning rate and map size
        :param learning_rate: The learning rate.
        :param batch_size: The batch size to use.
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
        all_activations = []

        # Make a batch generator.
        accumulator = np.zeros_like(self.weights)
        num_updates = 0

        num_batches = np.ceil(len(X) / batch_size).astype(int)

        for index in progressbar(range(num_batches), idx_interval=1, mult=batch_size):

            # Select the current batch.
            batch = X[index * batch_size: (index+1) * batch_size]

            update, differences = self._batch(batch, cache, map_radius_squared, learning_rate)

            all_activations.extend(np.sqrt(np.sum(np.square(differences), axis=2)))
            accumulator += update
            num_updates += 1

        self.weights += (accumulator / num_updates)

        return np.array(all_activations)

    def _batch(self, batch, cache, map_radius, learning_rate):

        bmus, differences = self._get_bmus(batch)

        influences = []

        for bmu in bmus:

            try:
                influence = cache[bmu]
            except KeyError:
                x_, y_ = self._index_dict[bmu]
                influence = self._calculate_influence(center_x=x_, center_y=y_, radius=map_radius)
                influence = np.tile(influence, (self.data_dim, 1)).T * learning_rate
                cache[bmu] = influence

            influences.append(influence)

        influences = np.array(influences)
        update = self._update(differences, influences).mean(axis=0)

        return update, differences

    def _update(self, input_vector, influence):
        """
        Updates the nodes, conditioned on the input vector,
        the influence, as calculated above, and the learning rate.

        :param input_vector: The input vector.
        :param influence: The influence the result has on each unit, depending on distance.
        """

        return influence * input_vector

    def _get_bmus(self, x):
        """
        Gets the best matching units, based on euclidean distance.

        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        differences = self._pseudo_distance(x, self.weights)
        distances = np.sqrt(np.sum(np.square(differences), axis=2))
        return np.argmin(distances, axis=1), differences

    def _pseudo_distance(self, X, weights):
        """
        Calculates the euclidean distance between an input and all the weights in range.

        :param x: The input.
        :param weights: An array of weights.
        :return: The distance from the input of each weight.
        """
        p = np.tile(X, (1, self.map_dim)).reshape((X.shape[0], self.map_dim, X.shape[1]))
        return p - weights

    def _calculate_influence(self, radius, **kwargs):
        """
        Finds the nearest neighbors, based on the current grid.
        see _create_grid.

        We only need to recalculate the grid every time we change the
        radius.

        :param center_x: An integer, representing the x coordinate
        :param center_y: An integer, representing the y coordinate
        :return: a tuple of indices and distances to the nodes at these indices.
        """

        # Add the current coordinates to the grid.
        temp_x = self.grid[0] + kwargs['center_x']
        temp_y = self.grid[1] + kwargs['center_y']

        x_cond = [np.logical_and(temp_x >= 0, temp_x < self.width)]
        y_cond = [np.logical_and(temp_y >= 0, temp_y < self.height)]
        mask = np.logical_and(x_cond, y_cond).ravel()

        temp_x = temp_x[mask]
        temp_y = temp_y[mask]
        distances = self.grid_distances[mask]

        indices = [self._coord_dict[x][y] for x, y in zip(temp_x, temp_y)]
        placeholder = np.zeros((self.map_dim,))

        # influence = np.tile(np.exp(-(distances ** 2 / (2 * radius ** 2)) * learning_rate, (self.data_dim, 1)).T
        influence = np.exp(-(distances / radius))
        placeholder[indices] += influence

        return placeholder

    def _distance_grid(self, radius):
        """
        Creates a grid for easy processing of nearest neighbor searches.

        The radius only changes once per epoch, and distances
        between nodes do not differ. Hence, there is no reason to calculate
        distances each time we want to know nearest neighbors to some node.

        Could be faster with aggressive caching.

        :param radius: The current radius.
        :return: The grid itself, and the distances for each grid.
        These are represented as a list of coordinates of things which are within distance,
        and a list with the same dimensionality, representing the distances.
        """

        # Radius never needs to be higher than the actual dimensionality
        radius = min(max(self.width, self.height), radius)

        # Cast to int for indexing
        radint = int(radius)

        # Define the vector which is added to the grid
        # We use squared euclidean distance, so we raise to the power of 2
        adder = np.power([abs(x) for x in range(-radint, radint+1)], 2)

        # Adder looks like this for radius 2
        # [4 1 0 1 4]

        # Double the radius + 1 is the size of the grid
        double = (int(radius) * 2) + 1

        grid = np.zeros((double, double))

        for index in range(double):
            grid[index, :] += adder
            grid[:, index] += adder

        # Grid now looks like (for radius 2):
        #
        # [[ 8.  5.  4.  5.  8.]
        #  [ 5.  2.  1.  2.  5.]
        #  [ 4.  1.  0.  1.  4.]
        #  [ 5.  2.  1.  2.  5.]
        #  [ 8.  5.  4.  5.  8.]]

        # We are doing euclidean distance, so sqrt
        grid = np.sqrt(grid)

        # Grid now looks like this:
        # [[ 2.82842712  2.23606798  2.          2.23606798  2.82842712]
        #  [ 2.23606798  1.41421356  1.          1.41421356  2.23606798]
        #  [ 2.          1.          0.          1.          2.        ]
        #  [ 2.23606798  1.41421356  1.          1.41421356  2.23606798]
        #  [ 2.82842712  2.23606798  2.          2.23606798  2.82842712]]

        where = np.where(grid < radius)

        return (np.array(where) - radint), np.square(grid[where])

    def predict(self, X):
        """
        Predicts node identity for input data.
        Similar to a clustering procedure.

        :param x: The input data.
        :return: A list of indices
        """

        # Return the indices of the BMU which matches the input data most
        bmus, _ = self._get_bmus(X)
        return bmus

    def predict_distance(self, X):

        _, differences = self._get_bmus(X)
        return np.sqrt(np.sum(np.square(differences), axis=2))

    def map_weights(self):
        """
        Retrieves the grid as a list of lists of weights. For easy visualization.

        :return: A three-dimensional Numpy array of values (width, height, data_dim)
        """

        mapped_weights = []

        for x in range(self.width):
            x *= self.height
            temp = []
            for y in range(self.height):
                temp.append(self.weights[x + y])

            mapped_weights.append(temp)

        return np.array(mapped_weights)

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

    s = Som(50, 50, 3, [1.0])
    start = time.time()
    bmus = s.train(colors, num_epochs=20)

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))

    '''from visualization.umatrix import UMatrixView

    view = UMatrixView(500, 500, 'dom')
    view.create(s.weights, colors, s.width, s.height, bmus[-1])
    view.save("junk_viz/_{0}.svg".format(0))

    print("Made {0}".format(0))'''