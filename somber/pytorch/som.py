import logging
import time
import numpy as np
import torch as t
import torch.nn as nn
# import torch.cuda as t
import json

from ..utils import expo, linear, progressbar
from functools import reduce
from collections import Counter, defaultdict


logger = logging.getLogger(__name__)


def euclidean(x, weights):
    """
    batched version of the euclidean distance.

    :param x: The input
    :param weights: The weights
    :return: A matrix containing the distance between each
    weight and each input.
    """
    m_norm = t.sum(t.pow(x, 2), dim=1)
    w_norm = t.sum(t.pow(weights, 2), dim=1)
    dotted = t.mm(t.mul(x, 2), weights.t())

    res = t.mm(m_norm, t.ones((1, w_norm.size()[0])))
    res += t.mm(t.ones((m_norm.size()[0], 1)), w_norm.t())
    res -= dotted

    return res


class Som(object):
    """
    This is the batched version of the basic SOM class.
    """

    def __init__(self,
                 map_dim,
                 data_dim,
                 learning_rate,
                 lrfunc=expo,
                 nbfunc=expo,
                 sigma=None,
                 min_max=t.min,
                 distance_function=euclidean,
                 influence_size=None):
        """
        A batched Self-Organizing-Map.

        :param map_dim: A tuple describing the MAP size.
        :param data_dim: The dimensionality of the input matrix.
        :param learning_rate: The learning rate.
        :param sigma: The neighborhood factor.
        :param lrfunc: The function used to decrease the learning rate.
        :param nbfunc: The function used to decrease the neighborhood
        :param min_max: The function used to determine the winner.
        :param distance_function: The function used to do distance calculation.
        Euclidean by default.
        :param influence_size: The size of the influence matrix.
        Usually reverts to data_dim, but can be
        larger.
        """
        if sigma is not None:
            self.sigma = sigma
        else:
            # Add small constant to sigma to prevent
            # divide by zero for maps of size 2.
            self.sigma = (max(map_dim) / 2.0) + 0.01

        self.distance_function = distance_function
        self.learning_rate = learning_rate
        # A tuple of dimensions
        # Usually (width, height), but can accomodate one-dimensional maps.
        self.map_dimensions = map_dim

        # The dimensionality of the weight vector
        # Usually (width * height)
        self.weight_dim = int(reduce(np.multiply, map_dim, 1))

        # Weights are initialized to small random values.
        # Initializing to more appropriate values given the dataset
        # will probably give faster convergence.
        self.weights = t.zeros(self.weight_dim, data_dim)
        self.data_dim = data_dim

        # The function used to diminish the learning rate.
        self.lrfunc = lrfunc
        # The function used to diminish the neighborhood size.
        self.nbfunc = nbfunc

        # Initialize the distance grid: only needs to be done once.
        self.distance_grid = self._initialize_distance_grid()
        self.min_max = min_max
        self.trained = False

        self.progressbar_interval = 10
        self.progressbar_mult = 1
        if influence_size is None:
            self.influence_size = self.data_dim
        else:
            self.influence_size = influence_size

    def train(self,
              X,
              num_epochs=10,
              total_updates=1000,
              stop_lr_updates=1.0,
              stop_nb_updates=1.0,
              batch_size=100,
              show_progressbar=False):
        """
        Fit the SOM to some data.

        The updates correspond to the number of updates to the parameters
        (i.e. learning rate, neighborhood, not weights!)
        to perform during training.

        In general, 1000 updates will do for most learning problems.

        :param X: the data on which to train.
        :param num_epochs: the number of epochs for which to train.
        :param total_updates: The number of updates to the parameters to do
        during training.
        :param stop_lr_updates: A fraction, describing over which portion of
        the training data the learning rate should decrease. If the total
        number of updates, for example is 1000, and stop_updates = 0.5,
        1000 updates will have occurred after half of the examples.
        After this period, no updates of the parameters will occur.
        :param stop_nb_updates: A fraction, describing over which portion of
        the training data the neighborhood should decrease.
        :param batch_size: the batch size
        :param show_progressbar: whether to show the progress bar.
        :return: None
        """
        if not self.trained:
            min_ = np.min(X, axis=0)
            random = np.random.rand(self.weight_dim).reshape((self.weight_dim, 1))
            temp = np.outer(random, np.abs(np.max(X, axis=0) - min_))
            self.weights = t.from_numpy(np.asarray(min_ + temp, dtype=np.float32))

        # The train length
        train_length = (len(X) * num_epochs) // batch_size

        X = self._create_batches(X, batch_size)
        X = t.from_numpy(np.asarray(X, dtype=np.float32))

        # The step size is the number of items between rough epochs.
        # We use len instead of shape because len also works with np.flatiter
        step_size_lr = max((train_length * stop_lr_updates) // total_updates, 1)
        step_size_nb = max((train_length * stop_nb_updates) // total_updates, 1)

        # Precalculate the number of updates.
        lr_update_counter = np.arange(step_size_lr,
                                      (train_length * stop_lr_updates) + step_size_lr,
                                      step_size_lr)

        nb_update_counter = np.arange(step_size_nb,
                                      (train_length * stop_nb_updates) + step_size_nb,
                                      step_size_nb)

        start = time.time()

        # Train
        nb_step = 0
        lr_step = 0
        idx = 0

        for epoch in range(num_epochs):

            if show_progressbar:
                logger.info("Epoch {0} of {1}".format(epoch, num_epochs))

            idx, nb_step, lr_step = self._epoch(X,
                                                nb_update_counter,
                                                lr_update_counter,
                                                idx, nb_step,
                                                lr_step,
                                                show_progressbar)

        self.trained = True

        logger.info("Total train time: {0}".format(time.time() - start))

    def _epoch(self,
               X,
               nb_update_counter,
               lr_update_counter,
               idx,
               nb_step,
               lr_step,
               show_progressbar):
        """
        A single epoch.

        This function uses an index parameter which is passed around to see to
        how many training items the SOM has been exposed globally.

        nb and lr_update_counter hold the indices at which the neighborhood
        size and learning rates need to be updated.
        These are therefore also passed around. The nb_step and lr_step
        parameters indicate how many times the neighborhood
        and learning rate parameters have been updated already.

        :param X: The training data.
        :param nb_update_counter: The indices at which to
        update the neighborhood.
        :param lr_update_counter: The indices at which to
        update the learning rate.
        :param idx: The current index.
        :param nb_step: The current neighborhood step.
        :param lr_step: The current learning rate step.
        :param show_progressbar: Whether to show a progress bar or not.
        :return: The index, neighborhood step and learning rate step
        """

        # Calculate the influences for update 0.
        map_radius = self.nbfunc(self.sigma,
                                 nb_step,
                                 len(nb_update_counter))

        learning_rate = self.lrfunc(self.learning_rate,
                                    lr_step,
                                    len(lr_update_counter))

        influences = self._calculate_influence(map_radius) * learning_rate

        # Iterate over the training data
        for x in progressbar(X, use=show_progressbar, logger=logger):

            self._example(x, influences)

            if idx in nb_update_counter:
                nb_step += 1

                map_radius = self.nbfunc(self.sigma,
                                         nb_step,
                                         len(nb_update_counter))
                logger.info("Updated map radius: {0}".format(map_radius))

                # The map radius has been updated, so the influence
                # needs to be recalculated
                influences = self._calculate_influence(map_radius)
                influences *= learning_rate

            if idx in lr_update_counter:
                lr_step += 1

                # Reset the influences back to 1
                influences /= learning_rate
                learning_rate = self.lrfunc(self.learning_rate,
                                            lr_step,
                                            len(lr_update_counter))
                logger.info("Updated learning rate: {0}".format(learning_rate))
                # Recalculate the influences
                influences *= learning_rate

        return idx, nb_step, lr_step

    def _create_batches(self, X, batch_size):
        """
        Create batches out of a sequence of data.

        Assumes ndim(X) == 2.

        This function will append zeros to the end of your data to ensure that
        all batches are even-sized.

        :param X: A numpy array, representing your input data.
        Must have 2 dimensions.
        :param batch_size: The desired batch size.
        :return: A batched version of your data.
        """
        self.progressbar_interval = 1
        self.progressbar_mult = batch_size

        max_x = int(np.ceil(X.shape[0] / batch_size))
        return np.resize(X, (max_x, batch_size, X.shape[1]))

    def _example(self, x, influences, **kwargs):
        """
        A single example.

        :param X: a numpy array of data, representing a single batch
        :param influences: The influence at the current epoch,
        given the learning rate and map size
        :return: An array representing the difference between
        weight vectors and the input.
        """
        activation, difference_x = self._get_bmus(x)
        influence, bmu = self._apply_influences(activation, influences)
        self.weights += self._calculate_update(difference_x, influence).mean(0)

        return activation

    def _distance_difference(self, x, weights):
        """
        Calculate the difference between an input and all the weights.

        :param x: The input.
        :param weights: An array of weights.
        :return: A vector of differences.
        """
        p = t.ones(1, weights.size()[0])
        return t.stack([v[:, None].mm(p).t() - weights for v in x])

    def _predict_base(self, X):
        """
        Predict activations to some input data.

        This function should not be used directly.

        :param X: The input data.
        :return: An array of arrays, representing the activation
        each node has to each input.
        """
        X = self._create_batches(X, 1)
        X = t.from_numpy(np.asarray(X, dtype=np.float32))

        activations = []

        for x in X:
            activation, _ = self._get_bmus(x)
            activations.extend(activation)

        return t.stack(activations)

    def _apply_influences(self, activations, influences):
        """
        Calculate the BMU using min_max, and get the appropriate influences.

        Then gets the appropriate influence from the neighborhood,
        given the BMU

        :param activations: A Numpy array of distances.
        :param influences: A (map_dim, map_dim, data_dim) array
        describing the influence each node has on each other node.
        :return: The influence given the bmu, and the index of the bmu itself.
        """
        bmu = self.min_max(activations, 1)[1].t()[0]
        return influences[bmu], bmu

    def _calculate_influence(self, sigma):
        """
        Pre-calculate the influence for a given value of sigma.

        The neighborhood has size map_dim * map_dim, so for a 30 * 30 map,
        the neighborhood will be size (900, 900). It is then duplicated
        influence_size times, and reshaped into an
        (map_dim, map_dim, influence_size) array.
        This is done to facilitate fast calculation in subsequent steps.

        :param sigma: The neighborhood value.
        :return: The neighborhood, reshaped into an array
        """
        neighborhood = t.exp(-self.distance_grid / (2.0 * sigma ** 2))
        # Reshape neighborhood
        neighborhood = neighborhood.view(self.weight_dim, self.weight_dim)
        # Repeat neighborhood self.influence_size times
        neighborhood = t.stack([neighborhood] * self.influence_size)

        # Awkward transpose
        return neighborhood.transpose(0, 2).transpose(0, 1)

    def _initialize_distance_grid(self):
        """
        Initialize the distance grid by calls to _grid_dist.

        :return:
        """
        p = [self._grid_distance(i).view(1, self.weight_dim) for i in range(self.weight_dim)]
        return t.stack(p)

    def _grid_distance(self, index):
        """
        Calculate the distance grid for a single index position.

        The distance grid is pre-calculated for fast neighborhood calculations
        later on (see _calc_influence).

        :param index: The index for which to calculate the distances.
        :return: A flattened version of the distance array.
        """
        width, height = self.map_dimensions

        row = index // width
        column = index % width

        # Fast way to construct distance matrix
        f = t.arange(0, self.weight_dim).view(self.map_dimensions)
        x = t.abs(f % width - row)
        y = t.abs(f % height - column).transpose(1, 0)

        distance = x + y

        # Return flattened version
        return distance.view(distance.numel())

    def map_weights(self):
        """
        Retrieve the grid as a list of lists of weights.

        :return: A three-dimensional Numpy array of values.
        """
        width, height = self.map_dimensions
        # Reshape to appropriate dimensions
        view = self.weights.view((width, height, self.data_dim))
        return np.array(view.transpose(1, 0).tolist())

    @classmethod
    def load(cls, path):
        """
        Loads a SOM from a JSON file.

        :param path: The path to the JSON file.
        :return: A SOM.
        """

        data = json.load(open(path))

        weights = data['weights']
        weights = t.from_numpy(np.asarray(weights, dtype=np.float32))
        datadim = weights.shape[1]

        dimensions = data['dimensions']
        lrfunc = expo if data['lrfunc'] == 'expo' else linear
        nbfunc = expo if data['nbfunc'] == 'expo' else linear
        lr = data['lr']
        sigma = data['sigma']

        s = cls(dimensions, datadim, lr, lrfunc=lrfunc, nbfunc=nbfunc, sigma=sigma)
        s.weights = weights
        s.trained = True

        return s

    def save(self, path):
        """
        Saves a SOM to a JSON file.

        :param path: The path to the JSON file that will be created
        :return: None
        """

        to_save = {}
        to_save['weights'] = [[float(w) for w in x] for x in self.weights]
        to_save['dimensions'] = self.map_dimensions
        to_save['lrfunc'] = 'expo' if self.lrfunc == expo else 'linear'
        to_save['nbfunc'] = 'expo' if self.nbfunc == expo else 'linear'
        to_save['lr'] = self.learning_rate
        to_save['sigma'] = self.sigma

        json.dump(to_save, open(path, 'w'))

    def quant_error(self, X):
        """
        Calculate the quantization error.

        Find the the minimum euclidean distance between the units and
        some input.

        :param X: Input data.
        :return: A vector of numbers, representing the quantization error
        for each data point.
        """
        dist = self._predict_base(X)
        return self.min_max(dist, 1)[0]

    def predict(self, X):
        """
        Predict the BMU for each input data.

        :param X: Input data.
        :return: The index of the bmu which best describes the input data.
        """
        dist = self._predict_base(X)
        return self.min_max(dist, 1)[1]

    def receptive_field(self, X, identities, max_len=5, threshold=0.9):
        """
        Calculate the receptive field of the SOM on some data.

        The receptive field is the common ending of all sequences which
        lead to the activation of a given BMU. If a SOM is well-tuned to
        specific sequences, it will have longer receptive fields, and therefore
        gives a better description of the dynamics of a given system.

        :param X: Input data.
        :param identities: The letters associated with each input datum.
        :param max_len: The maximum length sequence we expect.
        Increasing the window size leads to accurate results,
        but costs more memory.
        :return: The receptive field of each neuron.
        """
        assert len(X) == len(identities)

        receptive_fields = defaultdict(list)
        predictions = self.predict(X)

        for idx, p in enumerate(t.squeeze(predictions).tolist()):
            receptive_fields[p].append(identities[idx+1 - max_len:idx+1])

        sequence = defaultdict(list)

        for k, v in receptive_fields.items():

            v = [x for x in v if np.any(x)]

            total = len(v)
            v = ["".join([str(x_) for x_ in np.squeeze(x)]) for x in v]

            for row in reversed(list(zip(*v))):

                r = Counter(row)

                for _, count in r.items():
                    if count / total > threshold:
                        sequence[k].append(row[0])
                    else:
                        break

        return {k: v[::-1] for k, v in sequence.items()}

    def invert_projection(self, X, identities):
        """
        Calculate the inverted projection.

        The inverted projectio of a SOM is created by association each weight
        with the input datum which matches it the most.
        Works best for symbolic (instead of continuous) input data.

        :param X: Input data
        :param identities: The identities for each
        input datum, must be same length as X
        :return: A numpy array with identities, the shape of the map.
        """
        assert len(X) == len(identities)

        # Remove all duplicates from X
        X_unique, names = zip(*set([tuple((tuple(s), n)) for s, n in zip(X, identities)]))
        node_match = []

        X_unique = np.array(X_unique)

        for node in np.array(self.weights.tolist()):

            differences = node - X_unique
            distances = np.sum(np.square(differences), 1)
            node_match.append(names[np.argmin(distances)])

        return np.array(node_match)

    def _get_bmus(self, x):
        """
        Get the best matching units, based on euclidean distance.

        :param x: The input vector
        :return: The activations, and
        the differences between the input and the weights, which can be
        reused in the update calculation.
        """
        differences = self._distance_difference(x, self.weights)
        activations = self.distance_function(x, self.weights)
        return activations, differences

    def _calculate_update(self, difference_vector, influence):
        """
        Multiply the difference vector with the influence vector.

        The influence vector is chosen based on the BMU, so that the update
        done to the every weight node is proportional to its distance from the
        BMU.

        Implicitly uses Oja's Rule: delta_W = alpha * (X - w)

        In this case (X - w) has been precomputed for speed, in the function
        _get_bmus.

        :param difference_vector: The difference between the input and
        weights.
        :param influence: The influence the result has on each unit,
        depending on distance. Already includes the learning rate.
        """
        return difference_vector * influence
