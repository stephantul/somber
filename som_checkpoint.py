import numpy as np
import time
import logging
import cProfile

from progressbar import progressbar


logger = logging.getLogger(__name__)


class Som(object):

    def __init__(self, width, height, dim, learning_rate):

        self.scaling_factor = max(width, height) / 2
        self.scaler = 0
        self.learning_rate = learning_rate

        self.width = width
        self.height = height
        self.weights = np.random.normal(0, 0.1, size=(width * height, dim))
        self.grid = None
        self.grid_distances = None

        self._index_dict = {idx: (idx // self.height, idx % self.height) for idx in range(self.weights.shape[0])}
        self._coord_dict = {v: k for k, v in self._index_dict.items()}

        self.trained = False

    def fit(self, num_epochs, data, return_history=0):
        """
        Fits the SOM to some data for a number of epochs.
        As the learning rate is decreased proportionally to the number
        of epochs, incrementally training a SOM is not feasible.

        :param num_epochs: The number of epochs for which to train
        :param data: The data on which to train
        :param return_history:
        :return: None
        """

        # Scaler ensures that the neighborhood radius is 0 at the end of training
        # given a square map.
        self.scaler = num_epochs / np.log(self.scaling_factor)

        # Local copy of learning rate.
        learning_rate = self.learning_rate

        # First history
        history = [self.map_weights()]
        history_bmu = []
        prev_bmus = None

        for epoch in range(num_epochs):

            logger.info("Epoch {0}".format(epoch))

            start = time.time()

            # Calculate the radius to see which BMUs attract one another
            map_radius = self.scaling_factor * np.exp(-epoch / self.scaler)

            # Create the helper grid, which absolves the need for expensive
            # euclidean products
            self.grid, self.grid_distances = self._create_grid(map_radius)

            # Get the indices of the Best Matching Units given the data.
            # bmus = self._get_bmus(data, self.weights)

            if not prev_bmus:
                euclid = self._euclid(data, self.weights)
                prev_bmus = [np.argsort(probas)[:1000] for probas in euclid]
                bmus = [bmu[0] for bmu in prev_bmus]
            else:
                euclid = self._euclid(data, self.weights, prev_bmus)
                bmus = [np.argmin(bmu) for bmu in euclid]

            # Convert the indices of the BMUs to coordinates (x, y)
            coords = self._indices_to_coords(bmus)

            logger.info("Setup time: {0}".format(time.time() - start))

            start = time.time()

            for vbmu_idx in progressbar(zip(data, coords)):

                vector, bmu_idx = vbmu_idx
                x, y = bmu_idx

                # Look up which neighbors are close enough to influence
                indices, scores = self._find_neighbors(x, y)
                # Calculate the influence
                influence = self._calculate_influence(scores, map_radius)

                # Update all units which are in range
                self._update(vector, self.weights, indices, influence, learning_rate)

            else:
                logger.info("Training epoch {0}/{1} took {2:.2f} seconds".format(epoch, num_epochs, time.time() - start))

            # Update learning rate
            learning_rate = self.learning_rate * np.exp(-epoch/num_epochs)

            if epoch % return_history == 0:
                history.append((self.predict(data), self.weights))

        self.trained = True

        if return_history:
            return history, history_bmu

    def predict(self, x):
        """
        Predicts node identity for input data.
        Similar to a clustering procedure.

        :param x: The input data.
        :return: A list of indices
        """

        if not self.trained:
            raise ValueError("Not trained yet")

        # Return the indices of the BMU which matches the input data most
        return self._get_bmus(x, self.weights)

    def predict_pseudo_proba(self, x):

        if not self.trained:
            raise ValueError("Not trained yet")

        return dict(enumerate(self._euclid(x, self.weights)))

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

    def _find_neighbors(self, center_x, center_y):
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

        # Add the current coordinates to the grid.
        temp_x = self.grid[0] + center_x
        temp_y = self.grid[1] + center_y

        x_cond = [np.logical_and(temp_x >= 0, temp_x < self.width)]
        y_cond = [np.logical_and(temp_y >= 0, temp_y < self.height)]

        mask = np.logical_and(x_cond, y_cond).ravel()

        temp_x = temp_x[mask]
        temp_y = temp_y[mask]
        distances = self.grid_distances[mask]

        return self._coords_to_indices(zip(temp_x, temp_y)), distances

    def _create_grid(self, radius):
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
        learning rate, condition on distance

        :param distances: A vector of distances
        :param map_radius: The current radius
        :return: A vector of scores
        """

        return np.exp(-(distances ** 2 / map_radius ** 2))

    def _update(self, input_vector, weights, indices, influence, learning_rate):
        """
        Updates the nodes, conditioned on the input vector,
        the influence, as calculated above, and the learning rate.

        :return: None
        """

        if not len(indices):
            return

        influence = np.repeat(influence, input_vector.shape[0]).reshape(influence.shape[0], input_vector.shape[0])
        weights[indices] += influence * (learning_rate * (input_vector - weights[indices]))

        return weights

    def _get_bmus(self, x, weights):
        """
        Gets the best matching units, based on euclidean distance.

        :param x: The input vector
        :param weights: The weight vectors
        :return: A list of integers, representing the indices of the best matching units.
        """
        # return [np.argmax(v) for v in weights.dot(x.T)]
        return [np.argmin(probas) for probas in self._euclid(x, weights)]

    @staticmethod
    def _euclid(x, weights, bmu_histories=None):

        if bmu_histories:
            [np.linalg.norm(weights[bmu_history] - vector, axis=1) for bmu_history, vector in zip(bmu_histories, x)]
        return [np.linalg.norm(weights - vector, axis=1) for vector in x]

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    import pandas as pd

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
          [.66, .66, .66]])'''

    '''colors = []

    for x in range(10):
        for y in range(10):
            for z in range(10):
                colors.append((x/10, y/10, z/10))'''

    # colors = np.array(colors)

    color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'greyblue', 'lilac', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'darkgrey', 'mediumgrey', 'lightgrey']


    dlen = 700
    tetha = np.random.uniform(low=0,high=2*np.pi,size=dlen)[:,np.newaxis]
    X1 = 3*np.cos(tetha)+ .22*np.random.rand(dlen,1)
    Y1 = 3*np.sin(tetha)+ .22*np.random.rand(dlen,1)
    Data1 = np.concatenate((X1,Y1),axis=1)

    X2 = 1*np.cos(tetha)+ .22*np.random.rand(dlen,1)
    Y2 = 1*np.sin(tetha)+ .22*np.random.rand(dlen,1)
    Data2 = np.concatenate((X2,Y2),axis=1)

    X3 = 5*np.cos(tetha)+ .22*np.random.rand(dlen,1)
    Y3 = 5*np.sin(tetha)+ .22*np.random.rand(dlen,1)
    Data3 = np.concatenate((X3,Y3),axis=1)

    X4 = 8*np.cos(tetha)+ .22*np.random.rand(dlen,1)
    Y4 = 8*np.sin(tetha)+ .22*np.random.rand(dlen,1)
    Data4 = np.concatenate((X4,Y4),axis=1)

    DataCL2 = np.concatenate((Data1,Data2,Data3,Data4),axis=0)

    s = Som(20, 20, 2, 0.3)
    start = time.time()
    history = s.fit(400, DataCL2, return_history=10)

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))

    # import matplotlib.pyplot as plt
    p = s.predict(DataCL2)

    from visualization.umatrix import UMatrixView

    for idx, x_w in enumerate(history):

        x, weight = x_w

        view = UMatrixView(500, 500, 'dom')
        view.create(weight, DataCL2, s.width, s.height, x)
        view.save("_{0}.svg".format(x))