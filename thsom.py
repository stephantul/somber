import numpy as np
import time
import logging
import cProfile

from progressbar import progressbar
from preprocessing.pron_to_dictionary import pron_to_dictionary
from string import ascii_lowercase
from collections import OrderedDict
from itertools import  chain


logger = logging.getLogger(__name__)


class THSom(object):

    def __init__(self, embedding, indexes, width, height, dim, alpha, beta):

        self.embedding = np.array(embedding)
        self.indexes = indexes

        self.scaling_factor = max(width, height) / 2
        self.scaler = 0
        self.alpha = alpha
        self.beta = beta

        self.width = width
        self.height = height
        self.weights = np.random.normal(0, 1.0, size=(width * height, dim))
        self.temporal_weights = np.zeros((width * height, width * height))

        self.prev_activations = self.weights.sum(axis=1)
        self.prev_bmu = 0
        self.grid = None
        self.grid_distances = None

        self._index_dict = {idx: (idx // self.height, idx % self.height) for idx in range(self.weights.shape[0])}
        self._coord_dict = {v: k for k, v in self._index_dict.items()}
        self.const_dim = np.sqrt(self.weights.shape[-1])

        self.trained = False
        self.radii = {}

    def predict_sequence(self, sequence):

        bmus = []

        for x in sequence:
            bmu, res = self._get_bmus(x)
            self.prev_activations = res
            bmus.append(bmu)

        self.prev_activations *= 0

        return bmus

        # return list(zip(*[self._get_bmu(x) for x in sequence]))[0]

    def epoch_step(self, X, map_radius, alpha, beta):
        """
        pej

        :param X:
        :param map_radius:
        :param alpha:
        :param beta:
        :return:
        """

        '''prev_bmu = 0

        for x, bmu in zip(X, bmus):

            # Convert the indices of the BMUs to coordinates (x, y)
            coords = self._index_dict[bmu]

            # Look up which neighbors are close enough to influence
            indices, scores = self._find_neighbors(coords[0], coords[1])
            # Calculate the influence
            influence = self._calculate_influence(scores, map_radius)

            # Update all units which are in range
            self._update(x, indices, influence, alpha, beta, prev_bmu)

        return bmus'''

        # Create the helper grid, which absolves the need for expensive
        # euclidean products

        self.grid, self.grid_distances = self._distance_grid(map_radius)
        self.radii = {idx: self._find_neighbors(idx, map_radius) for idx in range(self.width * self.height)}

        temporal_sum = np.sum(self.temporal_weights, axis=1)
        weights_summed_squared = np.sum(np.power(self.weights, 2), axis=1)
        # Get the indices of the Best Matching Unit, given the data.

        activations = self._euclid(self.embedding, self.weights, weights_summed_squared).T
        activations /= activations.max(axis=1).reshape(activations.shape[0], 1)
        activations = np.vstack([activations, np.zeros((activations[-1].shape[0],))])
        activations_per_step = [activations[[-1] + indices] for indices in X]

        temporal = [self._temporal(step, temporal_sum) for step in activations_per_step]

        bmu_sequences = []

        for i, x, y in progressbar(zip(X, activations_per_step, temporal)):

            local_input = self.embedding[i]
            result = self.const_dim / (x + y)
            bmus = np.argmax(result, axis=1)

            bmu_sequences.append(bmus)

            neighborhoods, influences = zip(*[self.radii[bmu] for bmu in bmus])

            self._update(local_input, neighborhoods, influences, alpha, beta, bmus)

        return bmu_sequences

    def train(self, data, epochs=100000):
        """
        Fits the SOM to some data for a number of epochs.
        As the learning rate is decreased proportionally to the number
        of epochs, incrementally training a SOM is not feasible.

        :param data: The data on which to train.
        :param epochs: The number of samples to draw from the training data.
        :return: None
        """

        # Scaler ensures that the neighborhood radius is 0 at the end of training
        # given a square map.
        self.scaler = epochs / np.log(self.scaling_factor)

        # Local copy of learning rate.
        alpha = self.alpha
        beta = self.beta

        # First history

        for epoch in range(epochs):

            map_radius = self.scaling_factor * np.exp(-epoch / self.scaler)

            self.epoch_step(X, map_radius, alpha, beta)
            alpha = self.alpha * np.exp(-epoch / epochs)
            beta = self.beta * np.exp(-epoch / epochs)

        self.temporal_weights = np.array(self.temporal_weights)
        self.trained = True

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

    def _update(self, input_vector, neighborhood, influence, alpha, beta, bmus):
        """
        Updates the nodes, conditioned on the input vector,
        the influence, as calculated above, and the learning rate.

        """

        if not len(neighborhood):
            return

        prev_winner = 0

        for n, i, x, b in zip(neighborhood, influence, input_vector, bmus):

            influence_ = np.tile(i, (x.shape[0], 1)).T

            self.weights[n] += influence_ * (alpha * (x - self.weights[n]))

            # Temporal weights from the current winners to the previous winner are strengthened.
            # Temporal weights from the current non-winners to the previous winner are weakened
            # Temporal weights from the previous non-winners to the current neighborhood are weakened.

            temp = set(range(self.width * self.height))
            outside = np.array(list(temp - set(n)))
            losers = np.array(list(temp - {prev_winner}))

            # Strengthen Connections from winner to neighborhood
            self.temporal_weights[prev_winner, n] += i * (alpha * (1 - self.temporal_weights[prev_winner, n] + beta))

            # Weaken Connections from winner to outside-neighborhood
            self.temporal_weights[prev_winner, outside] -= (alpha * (self.temporal_weights[prev_winner, outside] + beta))

            # Weaken Connections from non-winners to neighborhood
            indices = np.ix_(losers, n)
            self.temporal_weights[indices] -= ((1 - i) * (alpha * (self.temporal_weights[indices] + beta)))

            prev_winner = b

        self.weights = self.weights.clip(min=0, max=1)
        self.temporal_weights = self.temporal_weights.clip(min=0, max=1)

    def _get_bmus(self, x, weights_squared):
        """
        Gets the best matching units, based on euclidean distance.

        :param x: The input vector
        :return: A list of integers, representing the indices of the best matching units.
        """

        act = self._euclid(x, self.weights, weights_squared)
        temp = self._temporal()

        # Regularization
        res = self.const_dim - (act + temp)

        bmu = np.argmax(res)
        # Normalization
        res /= res[bmu]

        # Use argmax because of regularization + normalization
        return bmu, res

    @staticmethod
    def _euclid(x, weights, squared):
        """
        euclidean distance

        :param x:
        :param weights:
        :return:
        """

        return np.dot(weights, x.T) * -2 + squared.reshape(squared.shape[0], 1)

    def _temporal(self, prev_activations, temporal_sum):
        """

        :return:
        """
        return np.sum(prev_activations * temporal_sum)

        # eq: y_i = sum(y_k_t-1 * weight(i, k))
        # prev_activations = all_y

    def _indices_to_coords(self, indices):
        """
        Helper function: converts a list of indices to a list of (x, y) coordinates,
        based on the width and height of the map.

        :param indices: A list of ints, representing the indices.
        :return: A list of tuples, (x, y).
        """

        return [self._index_dict[idx] for idx in indices]

    def _coords_to_indices(self, coords):
        """
        Helper function, converts a list of coordinates (x, y) to a list of indices.

        :param coords: A list of tuples, (x, y)
        :return: A list of indices.
        """

        return [self._coord_dict[tup] for tup in coords]

    def _find_neighbors(self, index, radius):
        """
        Finds the nearest neighbors, based on the current grid.
        see _create_grid.

        Simply put, the radius of the nearest neighbor search only changes once per epoch,
        and hence there is no need to calculate it for every node.
        So, we create a radius_grid, which we move around, depending on the
        coordinates of the current node.

        :param center_x: An integer, representing the x coordinate
        :param center_y: An integer, representing the y coordinate
        :return: a tuple of indices and distances to the nodes at these indices.
        """

        center_x, center_y = self._index_dict[index]

        # Add the current coordinates to the grid.
        temp_x = self.grid[0] + center_x
        temp_y = self.grid[1] + center_y

        x_cond = [np.logical_and(temp_x >= 0, temp_x < self.width)]
        y_cond = [np.logical_and(temp_y >= 0, temp_y < self.height)]

        mask = np.logical_and(x_cond, y_cond).ravel()

        temp_x = temp_x[mask]
        temp_y = temp_y[mask]
        distances = self.grid_distances[mask]

        return np.array(self._coords_to_indices(zip(temp_x, temp_y))), np.array(self._calculate_influence(distances, radius))

    def _distance_grid(self, radius):
        """
        Creates a grid for easy processing of nearest neighbor searches.

        As explained above, the radius only changes once per epoch, and distances
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

        return np.array(where) - radint, grid[where]

    @staticmethod
    def _calculate_influence(distances, map_radius):
        """
        Calculates influence, which can be described as a node-specific
        learning rate, conditioned on distance

        :param distances: A vector of distances
        :param map_radius: The current radius
        :return: A vector of scores
        """

        return np.exp(-(distances ** 2 / map_radius ** 2))


if __name__ == "__main__":

    import re
    reg = re.compile(r"\W")

    logging.basicConfig(level=logging.INFO)

    eye = np.eye(len(ascii_lowercase)+1)

    letters = {k: idx for idx, k in enumerate(ascii_lowercase)}
    letters['#'] = len(letters)

    wordlist = list(pron_to_dictionary("data/epl.cd", phon_indices=(11, 7)))
    wordlist = [reg.sub("", x) for x in wordlist]

    X = [np.array([letters[c] for c in word]) for word in wordlist]

    s = THSom(eye, X, width=35, height=35, dim=len(eye), alpha=0.3, beta=0.1)
    start = time.time()

    s.train(epochs=100, data=X)

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))
