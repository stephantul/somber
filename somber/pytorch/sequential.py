import numpy as np
import logging
import json
import torch as t

from .som import Som, euclidean
from ..utils import expo, progressbar, linear
from functools import reduce


logger = logging.getLogger(__name__)


class Sequential(Som):
    """
    A base class for sequential SOMs, removing some code duplication.

    Not usable as a stand-alone class
    """

    def __init__(self,
                 map_dim,
                 data_dim,
                 learning_rate,
                 sigma,
                 lrfunc=expo,
                 nbfunc=expo,
                 min_max=t.min,
                 distance_function=euclidean,
                 influence_size=None):
        """
        A base class for sequential SOMs, removing some code duplication.

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
        super().__init__(map_dim,
                         data_dim,
                         learning_rate,
                         sigma,
                         lrfunc,
                         nbfunc,
                         min_max,
                         distance_function,
                         influence_size=influence_size)

    def _init_prev(self, X):
        """
        A safe initialization for the first previous value.

        :param X: The input data.
        :return: A matrix of the appropriate size for simulating contexts.
        """
        return t.zeros((X.size()[1], self.weight_dim))

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
        # Initialize the previous activation
        prev_activation = self._init_prev(X)

        # Calculate the influences for update 0.
        map_radius = self.nbfunc(self.sigma,
                                 nb_step,
                                 len(nb_update_counter))

        learning_rate = self.lrfunc(self.learning_rate,
                                    lr_step,
                                    len(lr_update_counter))

        influences = self._calculate_influence(map_radius) * learning_rate

        # Iterate over the training data
        for x in progressbar(X,
                             use=show_progressbar,
                             mult=self.progressbar_mult,
                             idx_interval=self.progressbar_interval):

            prev_activation = self._example(x,
                                            influences,
                                            prev_activation=prev_activation)

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

            idx += 1

        return idx, nb_step, lr_step

    def _create_batches(self, X, batch_size):
        """
        Create subsequences out of a sequential piece of data.

        Assumes ndim(X) == 2.

        This function will append zeros to the end of your data to make
        sure all batches even-sized.

        :param X: A numpy array, representing your input data.
        Must have 2 dimensions.
        :param batch_size: The desired pytorch size.
        :return: A batched version of your data.
        """
        self.progressbar_interval = 1
        self.progressbar_mult = batch_size
        # This line first resizes the data to
        # (batch_size, len(X) / batch_size, data_dim)
        max_x = int(np.ceil(X.shape[0] / batch_size))
        X = np.resize(X, (batch_size, max_x, X.shape[1]))
        # Transposes it to (len(X) / batch_size, batch_size, data_dim)
        return X.transpose((1, 0, 2))

    def _get_bmus(self, x, **kwargs):

        pass

    def _predict_base(self, X):
        """
        Predict distances to some input data.

        :param X: The input data.
        :return: An array of arrays, representing the activation
        each node has to each input.
        """
        X = self._create_batches(X, 1)
        X = t.from_numpy(np.asarray(X, dtype=np.float32))

        activations = []

        prev = self._init_prev(X)

        for x in X:
            prev = self._get_bmus(x, prev_activation=prev)[0]
            activations.extend(prev)

        return t.stack(activations)


class Recurrent(Sequential):
    """A recurrent SOM."""

    def __init__(self,
                 map_dim,
                 data_dim,
                 learning_rate,
                 alpha,
                 sigma=None,
                 lrfunc=expo,
                 nbfunc=expo,
                 min_max=t.min,
                 distance_function=euclidean,
                 influence_size=None):
        """
        A recurrent SOM.

        The recurrent SOM attempts to model sequences by integrating
        the current weight vector with the activation in the previous
        time-step. Weights thus become shared between current and previous
        activation.

        :param map_dim: A tuple of map dimensions,
        e.g. (10, 10) instantiates a 10 by 10 map.
        :param data_dim: The data dimensionality.
        :param learning_rate: The learning rate, which is decreased
        according to some function.
        :param lrfunc: The function to use in decreasing the learning rate.
        The functions are defined in utils. Default is exponential.
        :param nbfunc: The function to use in decreasing the neighborhood size.
        The functions are defined in utils. Default is exponential.
        :param alpha: a float between 0 and 1, specifying how much weight the
        previous activation receives in comparison to the current activation.
        :param sigma: The starting value for the neighborhood size, which is
        decreased over time. If sigma is None (default), sigma is calculated as
        ((max(map_dim) / 2) + 0.01), which is generally a good value.
        """
        super().__init__(map_dim,
                         data_dim,
                         learning_rate,
                         lrfunc, nbfunc,
                         sigma, min_max,
                         distance_function)

        self.alpha = alpha

    def _init_prev(self, X):
        """
        Initialize the context vector.

        Initializes the context to a safe value at the beginning
        of training. In this case, this is a (batch * weight_dim * data_dim)
        matrix.

        :param X: The input data.
        :return: A (batch * weight_dim * data_dim) matrix.
        """
        return t.zeros(X.size()[1], self.weight_dim, X.size()[2])

    def _example(self, x, influences, **kwargs):
        """
        A single example.

        :param X: a numpy array of data, representing a single batch
        :param influences: The influence at the current epoch,
        given the learning rate and map size
        :return: An array representing the difference between
        weight vectors and the input.
        """
        prev = kwargs['prev_activation']

        # Get the indices of the Best Matching Units, given the data.
        activation, difference = self._get_bmus(x, prev_activation=prev)
        influence, bmu = self._apply_influences(activation, influences)
        self.weights += self._calculate_update(difference, influence).mean(0)

        return difference

    def _get_bmus(self, x, **kwargs):
        """
        Get the best matching units, based on euclidean distance.

        :param x: A batch of data.
        :return: The activation, and difference between the input and weights.
        """
        # Difference_x is the components of the weights
        # subtracted from the weight vector.
        difference_x = self._distance_difference(x, self.weights)
        distances = (1 - self.alpha) * kwargs['prev_activation'] + (self.alpha * difference_x)

        # Compute the actual activation through euclidean.
        activation = t.squeeze(t.stack([t.sum(t.pow(d, 2), 1) for d in distances]), 2)
        return activation, difference_x

    def _predict_base(self, X):
        """
        Predict distances to some input data.

        :param X: The input data.
        :return: An array of arrays, representing the activation
        each node has to each input.
        """
        X = self._create_batches(X, 1)
        X = t.from_numpy(np.asarray(X, dtype=np.float32))

        activations = []

        prev = self._init_prev(X)

        for x in X:
            prev_activation, prev = self._get_bmus(x, prev_activation=prev)
            activations.extend(prev_activation)

        return t.stack(activations)


class Recursive(Sequential):

    def __init__(self,
                 map_dim,
                 data_dim,
                 learning_rate,
                 alpha,
                 beta,
                 sigma=None,
                 lrfunc=expo,
                 nbfunc=expo):
        """
        A recursive SOM.

        A recursive SOM models sequences through context dependence by not only
        storing the exemplars in weights, but also storing which exemplars
        preceded them. Because of this organization, the SOM can recursively
        "remember" short sequences, which makes it attractive for simple
        sequence problems, e.g. characters or words.

        :param map_dim: A tuple of map dimensions,
        e.g. (10, 10) instantiates a 10 by 10 map.
        :param data_dim: The data dimensionality.
        :param learning_rate: The learning rate, which is decreased
        according to some function.
        :param lrfunc: The function to use in decreasing the learning rate.
        The functions are defined in utils. Default is exponential.
        :param nbfunc: The function to use in decreasing the neighborhood size.
        The functions are defined in utils. Default is exponential.
        :param alpha: a float value, specifying how much weight the
        input value receives in the BMU calculation.
        :param beta: a float value, specifying how much weight the context
        receives in the BMU calculation.
        :param sigma: The starting value for the neighborhood size, which is
        decreased over time. If sigma is None (default), sigma is calculated as
        ((max(map_dim) / 2) + 0.01), which is generally a good value.
        """
        super().__init__(map_dim,
                         data_dim,
                         learning_rate,
                         lrfunc,
                         nbfunc,
                         sigma,
                         min_max=t.max,
                         influence_size=reduce(np.multiply, map_dim))

        self.context_weights = t.zeros((self.weight_dim, self.weight_dim))
        self.alpha = alpha
        self.beta = beta

        self.context_weights = self.context_weights
        self.alpha = self.alpha
        self.beta = self.beta

    def _example(self, x, influences, **kwargs):
        """
        A single example.

        :param X: a numpy array of data
        :param influences: The influence at the current epoch,
        given the learning rate and map size
        :return: A vector describing activation values for each unit.
        """
        prev = kwargs['prev_activation']

        activation, diff_x, diff_context = self._get_bmus(x,
                                                          prev_activation=prev)

        influence, bmu = self._apply_influences(activation, influences)
        # Update
        self.weights += self._calculate_update(diff_x, influence[:, :, :self.data_dim]).mean(0)
        res = self._calculate_update(diff_context, influence).mean(0)
        self.context_weights += t.squeeze(res)

        return activation

    def _get_bmus(self, x, **kwargs):
        """
        Get the best matching units, based on euclidean distance.

        The euclidean distance between the context vector and context weights
        and input vector and weights are used to estimate the BMU. The
        activation of the units is the sum of the distances, weighed by two
        constants, alpha and beta.

        The exponent of the negative of this value describes the activation
        of the units. This function is bounded between 0 and 1, where 1 means
        the unit matches well and 0 means the unit doesn't match at all.

        :param x: A batch of data.
        :return: The activation, and difference between the input and weights.
        """
        prev_activation = kwargs['prev_activation']
        # Differences is the components of the weights subtracted
        # from the weight vector.
        difference_x = self._distance_difference(x, self.weights)
        difference_y = self._distance_difference(prev_activation,
                                                 self.context_weights)

        distance_x = self.distance_function(x, self.weights)
        distance_y = self.distance_function(prev_activation,
                                            self.context_weights)

        activation = t.exp(-(t.mul(distance_x, self.alpha) + t.mul(distance_y, self.beta)))

        return activation, difference_x, difference_y

    @classmethod
    def load(cls, path):
        """
        Load a recursive SOM from a JSON file.

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

        s = cls(dimensions,
                datadim,
                lr,
                lrfunc=lrfunc,
                nbfunc=nbfunc,
                sigma=sigma,
                alpha=alpha,
                beta=beta)

        s.weights = weights
        s.context_weights = context_weights
        s.trained = True

        return s

    def save(self, path):
        """
        Save a SOM to a JSON file.

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


class Merging(Sequential):
    """A merging som."""

    def __init__(self,
                 map_dim,
                 data_dim,
                 learning_rate,
                 alpha,
                 beta,
                 sigma=None,
                 lrfunc=expo,
                 nbfunc=expo,
                 min_max=t.min,
                 distance_function=euclidean):
        """
        A merging som.

        :param map_dim: A tuple of map dimensions,
        e.g. (10, 10) instantiates a 10 by 10 map.
        :param data_dim: The data dimensionality.
        :param learning_rate: The learning rate, which is decreased
        according to some function.
        :param lrfunc: The function to use in decreasing the learning rate.
        The functions are defined in utils. Default is exponential.
        :param nbfunc: The function to use in decreasing the neighborhood size.
        The functions are defined in utils. Default is exponential.
        :param alpha: Controls the rate of context dependence, where 0 is low
        context dependence, and 1 is high context dependence. Should start at
        low values (e.g. 0.0 to 0.05)
        :param beta: A float between 1 and 0 specifying the influence of
        context on previous weights. Static, usually 0.5.
        :param sigma: The starting value for the neighborhood size, which is
        decreased over time. If sigma is None (default), sigma is calculated as
        ((max(map_dim) / 2) + 0.01), which is generally a good value.
        """
        super().__init__(map_dim,
                         data_dim,
                         learning_rate,
                         lrfunc,
                         nbfunc,
                         sigma,
                         min_max,
                         distance_function)

        self.alpha = alpha
        self.beta = beta
        self.context_weights = t.ones(self.weights.size())
        self.entropy = 0

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
        map_radius = self.nbfunc(self.sigma, nb_step, len(nb_update_counter))
        learning_rate = self.lrfunc(self.learning_rate,
                                    lr_step,
                                    len(lr_update_counter))
        influences = self._calculate_influence(map_radius) * learning_rate

        prev = self._init_prev(X)

        # Iterate over the training data
        for x in progressbar(X,
                             use=show_progressbar,
                             mult=self.progressbar_mult,
                             idx_interval=self.progressbar_interval):

            prev = self._example(x,
                                 influences,
                                 prev_activation=prev)

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

            idx += 1

        return idx, nb_step, lr_step

    def _example(self, x, influences, **kwargs):
        """
        A single example.

        :param X: a numpy array of data
        :param influences: The influence at the current epoch,
        given the learning rate and map size
        :return: A vector describing activation values for each unit.
        """
        prev_bmu = self.min_max(kwargs['prev_activation'], 1)[1].t()[0]
        context = (1 - self.beta) * self.weights[prev_bmu] + self.beta * self.context_weights[prev_bmu]

        # Get the indices of the Best Matching Units, given the data.
        activation, diff_x, diff_context = self._get_bmus(x, prev_activation=context)
        influence, bmu = self._apply_influences(activation, influences)

        self.weights += t.mean(self._calculate_update(diff_x, influence), 0)
        self.context_weights += t.mean(self._calculate_update(diff_context, influence), 0)

        return activation

    def _entropy(self, prev_bmus, prev_update):
        """
        Calculate the entropy activation pattern.

        Merging SOMS perform better when their weight-based activation profile
        has high entropy, as small changes in context will then be able to have
        a larger effect.

        This is reflected in this function, which increases the importance of
        context by decreasing alpha if the entropy decreases. The function uses
        a very large momentum term of 0.9 to make sure the entropy does not
        rise or fall too sharply.

        :param prev_bmus: The previous BMUs.
        :param prev_update: The previous update, used as a momentum term.
        :return:
        """
        prev_bmus = np.array(list(prev_bmus.values()))
        prev_bmus = prev_bmus / np.sum(prev_bmus)

        new_entropy = -np.sum(prev_bmus * np.nan_to_num(np.log2(prev_bmus)))
        entropy_diff = (new_entropy - self.entropy)

        update = (entropy_diff * 0.1) + (prev_update * 0.9)

        self.entropy = new_entropy

        logger.info("Entropy: {0}".format(new_entropy))

        return update

    def _get_bmus(self, x, **kwargs):
        """
        Get the best matching units, based on euclidean distance.

        :param x: The input vector
        :return: An integer, representing the index of the best matching unit.
        """
        # Differences is the components of the weights
        # subtracted from the weight vector.
        differences_x = self._distance_difference(x, self.weights)
        # Idem for context.
        differences_y = self._distance_difference(kwargs['prev_activation'],
                                                  self.context_weights)

        distances_x = self.distance_function(x, self.weights)
        distances_y = self.distance_function(kwargs['prev_activation'],
                                             self.context_weights)

        # BMU is based on a weighted addition of current and
        # previous activation.
        activations = t.squeeze((t.mul(distances_x, 1 - self.alpha) + t.mul(distances_y, self.alpha)), 1)

        return activations, differences_x, differences_y

    def _predict_base(self, X):
        """
        Predict distances to some input data.

        :param X: The input data.
        :return: An array of arrays, representing the activation
        each node has to each input.
        """
        X = self._create_batches(X, 1)
        X = t.from_numpy(np.asarray(X, dtype=np.float32))

        distances = []

        prev_activation = self._init_prev(X)

        for x in X:
            prev_activation = self.weights[self.min_max(prev_activation, 1)[1].t()[0]]
            prev_activation = self._get_bmus(x, prev_activation=prev_activation)[0]
            distances.extend(prev_activation)

        return t.stack(distances)

    @classmethod
    def load(cls, path):
        """
        Load a SOM from a JSON file.

        A normal SOM can be loaded via this method. Any attributes not present
        in the loaded JSON will be initialized to sane values.

        :param path: The path to the JSON file.
        :return: A trained mergeSom.
        """
        data = json.load(open(path))

        weights = data['weights']
        weights = t.from_numpy(np.array(weights, dtype=np.float32))

        datadim = weights.shape[1]
        dimensions = data['dimensions']

        lrfunc = expo if data['lrfunc'] == 'expo' else linear
        nbfunc = expo if data['nbfunc'] == 'expo' else linear
        lr = data['lr']
        sigma = data['sigma']

        try:
            context_weights = data['context_weights']
            context_weights = t.from_numpy(np.array(context_weights, dtype=np.float32))
        except KeyError:
            context_weights = t.ones(weights.shape)

        try:
            alpha = data['alpha']
            beta = data['beta']
            entropy = data['entropy']
        except KeyError:
            alpha = 0.0
            beta = 0.5
            entropy = 0.0

        s = cls(dimensions,
                datadim,
                lr,
                lrfunc=lrfunc,
                nbfunc=nbfunc,
                sigma=sigma,
                alpha=alpha,
                beta=beta)

        s.entropy = entropy
        s.weights = weights
        s.context_weights = context_weights
        s.trained = True

        return s

    def save(self, path):
        """
        Save the merging SOM to a JSON file.

        :param path: The path to which to save the JSON file.
        :return: None
        """
        to_save = {}
        to_save['weights'] = [[float(w) for w in x] for x in self.weights]
        to_save['context_weights'] = [[float(w) for w in x] for x in self.context_weights]
        to_save['dimensions'] = self.map_dimensions
        to_save['lrfunc'] = 'expo' if self.lrfunc == expo else 'linear'
        to_save['nbfunc'] = 'expo' if self.nbfunc == expo else 'linear'
        to_save['lr'] = self.learning_rate
        to_save['sigma'] = self.sigma
        to_save['alpha'] = self.alpha
        to_save['beta'] = self.beta
        to_save['entropy'] = self.entropy

        json.dump(to_save, open(path, 'w'))
