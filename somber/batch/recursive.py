import numpy as np
import logging
import json
import torch as t

from somber.batch.som import Som
from somber.utils import expo, progressbar, linear


logger = logging.getLogger(__name__)


class Recursive(Som):

    def __init__(self, map_dim, data_dim, learning_rate, alpha, beta, sigma=None, lrfunc=expo, nbfunc=expo):
        """
        A recursive SOM.

        A recursive SOM models sequences through context dependence by not only storing the exemplars in weights,
        but also storing which exemplars preceded them. Because of this organization, the SOM can recursively
        "remember" short sequences, which makes it attractive for simple sequence problems, e.g. characters or words.

        :param map_dim: A tuple of map dimensions, e.g. (10, 10) instantiates a 10 by 10 map.
        :param data_dim: The data dimensionality.
        :param learning_rate: The learning rate, which is decreases according to some function
        :param lrfunc: The function to use in decreasing the learning rate. The functions are
        defined in utils. Default is exponential.
        :param nbfunc: The function to use in decreasing the neighborhood size. The functions
        are defined in utils. Default is exponential.
        :param alpha: The influence of the weight vector on the BMU decision
        :param beta: The influence of the context vector on the BMU decision
        :param sigma: The starting value for the neighborhood size, which is decreased over time.
        If sigma is None (default), sigma is calculated as ((max(map_dim) / 2) + 0.01), which is
        generally a good value.
        """

        super().__init__(map_dim, data_dim, learning_rate, lrfunc, nbfunc, sigma, min_max=t.max)
        self.context_weights = t.zeros((self.weight_dim, self.weight_dim))
        self.alpha = alpha
        self.beta = beta

        self.context_weights = self.context_weights
        self.alpha = self.alpha
        self.beta = self.beta

    def _epoch(self, X, nb_update_counter, lr_update_counter, idx, nb_step, lr_step, show_progressbar, context_mask):
        """
        A single epoch.

        :param X: The training data.
        :param nb_update_counter: The epochs at which to update the neighborhood.
        :param lr_update_counter: The epochs at which to updat the learning rate.
        :param idx: The current index.
        :param nb_step: The current neighborhood step.
        :param lr_step: The current learning rate step.
        :param show_progressbar: Whether to show a progress bar or not.
        :param context_mask: The context mask.
        :return:
        """

        prev_activation = t.zeros((X.size()[1], self.weight_dim))

        # Calculate the influences for update 0.
        map_radius = self.nbfunc(self.sigma, nb_step, len(nb_update_counter))
        learning_rate = self.lrfunc(self.learning_rate, lr_step, len(lr_update_counter))
        influences = self._calculate_influence(map_radius) * learning_rate
        update = False

        for x, ct in progressbar(zip(X, context_mask), use=show_progressbar):

            prev_activation = self._example(x, influences, prev_activation=prev_activation)
            # prev_activation *= ct

            if idx in nb_update_counter:
                nb_step += 1

                map_radius = self.nbfunc(self.sigma, nb_step, len(nb_update_counter))
                logger.info("Updated map radius: {0}".format(map_radius))
                update = True

            if idx in lr_update_counter:
                lr_step += 1

                learning_rate = self.lrfunc(self.learning_rate, lr_step, len(lr_update_counter))
                logger.info("Updated learning rate: {0}".format(learning_rate))
                update = True

            if update:
                influences = self._calculate_influence(map_radius) * learning_rate
                update = False

            idx += 1

        return idx, nb_step, lr_step

    def _example(self, x, influences, **kwargs):
        """
        A single example.

        :param X: a numpy array of data
        :param map_radius: The radius at the current epoch, given the learning rate and map size
        :param learning_rates: The learning rate.
        :param batch_size: The batch size
        :return: The best matching unit
        """

        prev_activation = kwargs['prev_activation']

        activation, diff_x, diff_context = self._get_bmus(x, prev_activation=prev_activation)

        influence, bmu = self._apply_influences(activation, influences)
        # Update
        self.weights += t.mean(self._calculate_update(diff_x, influence[:, :, :self.data_dim]), 0)
        res = t.squeeze(t.mean(self._calculate_update(diff_context, influence), 0))
        self.context_weights += res

        return activation

    def _create_batches(self, X, batch_size):
        """
        Creates batches out of a sequential piece of data.
        Assumes ndim(X) == 2.

        This function will append zeros to the end of your data to make all batches even-sized.

        For the recursive SOM, this function does not simply resize your data. It will create
        subsequences.

        :param X: A numpy array, representing your input data. Must have 2 dimensions.
        :param batch_size: The desired batch size.
        :return: A batched version of your data.
        """

        # This line first resizes the data to (batch_size, len(X) / batch_size, data_dim)
        X = np.resize(X, (batch_size, int(np.ceil(X.shape[0] / batch_size)), X.shape[1]))
        # Transposes it to (len(X) / batch_size, batch_size, data_dim)
        return X.transpose((1, 0, 2))

    def _get_bmus(self, x, **kwargs):
        """
        Gets the best matching units, based on euclidean distance.
        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """

        prev_activation = kwargs['prev_activation']
        # Differences is the components of the weights subtracted from the weight vector.
        difference_x = self._distance_difference(x, self.weights)
        difference_y = self._distance_difference(prev_activation, self.context_weights)

        # Distances are squared euclidean norm of differences.
        # Since euclidean norm is sqrt(sum(square(x)))) we can leave out the sqrt
        # and avoid doing an extra square.

        distance_x = self.distance_function(x, self.weights)
        distance_y = self.distance_function(prev_activation, self.context_weights)

        activation = t.exp(-(t.mul(distance_x, self.alpha) + t.mul(distance_y, self.beta)))

        return activation, difference_x, difference_y

    def _predict_base(self, X):
        """
        Predicts distances to some input data.

        :param X: The input data.
        :return: An array of arrays, representing the activation
        each node has to each input.
        """

        X = self._create_batches(X, 1)
        X = t.from_numpy(np.asarray(X, dtype=np.float32))
        print(X.size())

        distances = []

        prev_activation = np.squeeze(t.sum(t.pow(self._distance_difference(X[0], self.weights), 2), 2))[None, :]
        distances.extend(prev_activation)

        print(prev_activation.size())

        for x in X[1:]:
            prev_activation, _, _ = self._get_bmus(x, prev_activation=prev_activation)
            distances.extend(prev_activation)

        return t.stack(distances)

    @classmethod
    def load(cls, path):
        """
        Loads a recursive SOM from a JSON file.
        You can use this function to load weights of other SOMs.
        If there are no context weights, the context weights will be set to 0.

        :param path: The path to the JSON file.
        :return: A RecSOM.
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

        try:
            context_weights = data['context_weights']
            context_weights = t.from_numpy(np.asarray(context_weights, dtype=np.float32))
        except KeyError:
            context_weights = t.zeros((len(weights), len(weights)))

        try:
            alpha = data['alpha']
            beta = data['beta']
        except KeyError:
            alpha = 3.0
            beta = 1.0

        s = cls(dimensions, datadim, lr, lrfunc=lrfunc, nbfunc=nbfunc, sigma=sigma, alpha=alpha, beta=beta)
        s.weights = weights
        s.context_weights = context_weights
        s.trained = True

        return s

    def save(self, path):
        """
        Saves a SOM to a JSON file.

        :param path: The path to the JSON file that will be created
        :return: None
        """

        dicto = {}
        dicto['weights'] = [[float(w) for w in x] for x in self.weights]
        dicto['context_weights'] = [[float(w) for w in x] for x in self.context_weights]
        dicto['dimensions'] = self.map_dimensions
        dicto['lrfunc'] = 'expo' if self.lrfunc == expo else 'linear'
        dicto['nbfunc'] = 'expo' if self.nbfunc == expo else 'linear'
        dicto['lr'] = self.learning_rate
        dicto['sigma'] = self.sigma
        dicto['alpha'] = self.alpha
        dicto['beta'] = self.beta

        json.dump(dicto, open(path, 'w'))

    def _calculate_influence(self, sigma):
        """
        Pre-calculates the influence for a given value of sigma.

        The neighborhood has size map_dim * map_dim, so for a 30 * 30 map, the neighborhood will be
        size (900, 900). It is then duplicated _data_dim times, and reshaped into an
        (map_dim, map_dim, data_dim) array. This is done to facilitate fast calculation in subsequent steps.

        :param sigma: The neighborhood value.
        :return: The neighborhood, reshaped into an array
        """

        neighborhood = t.exp(-self.distance_grid / (2.0 * sigma ** 2)).view(self.weight_dim, self.weight_dim)
        return t.stack([neighborhood] * self.weight_dim).transpose(0,2).transpose(0,1)