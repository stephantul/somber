"""Base class for SOM and Neural Gas."""
import numpy as np
import logging
import time
import types
import json

from tqdm import tqdm
from .components.utilities import shuffle
from .components.initializers import range_initialization
from collections import Counter, defaultdict


logger = logging.getLogger(__name__)


class Base(object):
    """
    This is a base class for the Neural gas and SOM.

    parameters
    ==========
    num_neurons : int
        The number of neurons to create.
    data_dimensionality : int
        The dimensionality of the input data.
    params : dict
        A dictionary describing the parameters which need to be reduced
        over time. Each parameter is denoted by two fields: "value" and
        "factor", which denote the current value, and the constant factor
        with which the value is multiplied in each update step.
    argfunc : str, optional, default "argmin"
        The name of the function which is used for calculating the index of
        the BMU.
    valfunc : str, optional, default "min"
        The name of the function which is used for calculating the value of the
        BMU.
    initializer : function, optional, default range_initialization
        A function which takes in the input data and weight matrix and returns
        an initialized weight matrix. The initializers are defined in
        somber.components.initializers. Can be set to None.
    scaler : initialized Scaler instance, optional default None
        An initialized instance of Scaler() which is used to scale the data
        to have mean 0 and stdev 1. If this is set to None, the SOM will
        create a scaler.

    attributes
    ==========
    trained : bool
        Whether the som has been trained.
    weights : numpy array
        The weight matrix.
    param_names : set
        The parameter names. Used in saving.

    """

    # Static property names
    param_names = {'num_neurons',
                   'weights',
                   'data_dimensionality',
                   'params',
                   'valfunc',
                   'argfunc'}

    def __init__(self,
                 num_neurons,
                 data_dimensionality,
                 params,
                 argfunc="argmin",
                 valfunc="min",
                 initializer=range_initialization,
                 scaler=None):
        """Organize nothing."""
        self.num_neurons = np.int(num_neurons)
        self.data_dimensionality = data_dimensionality
        self.weights = np.zeros((num_neurons, data_dimensionality))
        self.argfunc = argfunc
        self.valfunc = valfunc
        self.trained = False
        self.scaler = scaler
        self.initializer = initializer
        self.params = params
        self.scaler = scaler

    def fit(self,
            X,
            num_epochs=10,
            updates_epoch=10,
            stop_param_updates=dict(),
            batch_size=1,
            show_progressbar=False):
        """
        Fit the learner to some data.

        parameters
        ==========
        X : numpy array.
            The input data.
        num_epochs : int, optional, default 10
            The number of epochs to train for.
        updates_epoch : int, optional, default 10
            The number of updates to perform on the learning rate and
            neighborhood per epoch. 10 suffices for most problems.
        stop_param_updates : dict
            The epoch at which to stop updating each param. This means
            that the specified parameter will be reduced to 0 at the specified
            epoch.
        batch_size : int, optional, default 100
            The batch size to use. Warning: batching can change your
            performance dramatically, depending on the task.
        show_progressbar : bool, optional, default False
            Whether to show a progressbar during training.

        """
        self._check_input(X)

        constants, X = self._pre_train(X,
                                       stop_param_updates,
                                       num_epochs,
                                       updates_epoch)

        start = time.time()

        for epoch in range(num_epochs):

            logger.info("Epoch {0} of {1}".format(epoch, num_epochs))

            self._epoch(X,
                        epoch,
                        batch_size,
                        updates_epoch,
                        constants,
                        show_progressbar)

        self.trained = True
        if self.scaler is not None:
            self.weights = self.scaler.inverse_transform(self.weights)

        logger.info("Total train time: {0}".format(time.time() - start))

    def _pre_train(self,
                   X,
                   stop_param_updates,
                   num_epochs,
                   updates_epoch):
        """Set parameters and constants before training."""
        X = np.asarray(X, dtype=np.float32)

        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        if self.initializer is not None:
            self.weights = self.initializer(X, self.weights)

        for v in self.params.values():
            v['value'] = v['orig']

        # Calculate the total number of updates given early stopping.
        updates = {k: stop_param_updates.get(k, num_epochs) * updates_epoch
                   for k, v in self.params.items()}

        # Calculate the value of a single step given the number of allowed
        # updates.
        single_steps = {k: np.exp(-((1.0 - (1.0 / v)))
                        * self.params[k]['factor'])
                        for k, v in updates.items()}

        # Calculate the factor given the true factor and the value of a
        # single step.
        constants = {k: np.exp(-self.params[k]['factor']) / v
                     for k, v in single_steps.items()}

        return constants, X

    def fit_predict(self,
                    X,
                    num_epochs=10,
                    updates_epoch=10,
                    stop_param_updates=dict(),
                    batch_size=1,
                    show_progressbar=False):
        """First fit, then predict."""
        self.fit(X,
                 num_epochs,
                 updates_epoch,
                 stop_param_updates,
                 batch_size,
                 show_progressbar)

        return self.predict(X, batch_size=batch_size)

    def fit_transform(self,
                      X,
                      num_epochs=10,
                      updates_epoch=10,
                      stop_param_updates=dict(),
                      batch_size=1,
                      show_progressbar=False):
        """First fit, then transform."""
        self.fit(X,
                 num_epochs,
                 updates_epoch,
                 stop_param_updates,
                 batch_size,
                 show_progressbar)

        return self.transform(X, batch_size=batch_size)

    def _epoch(self,
               X,
               epoch_idx,
               batch_size,
               updates_epoch,
               constants,
               show_progressbar):
        """
        Run a single epoch.

        This function shuffles the data internally,
        as this improves performance.

        parameters
        ==========
        X : numpy array
            The training data.
        epoch_idx : int
            The current epoch
        batch_size : int
            The batch size
        updates_epoch : int
            The number of updates to perform per epoch
        nb_constant : float
            The number to multiply the neighborhood with at every update step.
        lr_constant : float
            The number to multiply the learning rate with at every update step.
        show_progressbar : bool
            Whether to show a progressbar during training.

        """
        # Create batches
        X_ = self._create_batches(X, batch_size)
        X_len = np.prod(X.shape[:-1])

        update_step = np.ceil(X_.shape[0] / updates_epoch)

        # Initialize the previous activation
        prev_activation = self._init_prev(X_)
        influences = self._update_params(constants)

        # Iterate over the training data
        for idx, x in enumerate(tqdm(X_, disable=not show_progressbar)):

            # Our batches are padded, so we need to
            # make sure we know when we hit the padding
            # so we don't inadvertently learn zeroes.
            diff = X_len - (idx * batch_size)
            if diff and diff < batch_size:
                x = x[:diff]
                # Prev_activation may be None
                if prev_activation is not None:
                    prev_activation = prev_activation[:diff]

            # If we hit an update step, perform an update.
            if idx % update_step == 0:
                influences = self._update_params(constants)
                logger.info(self.params)

            prev_activation = self._propagate(x,
                                              influences,
                                              prev_activation=prev_activation)

    def _update_params(self, constants):
        """Update params and return new influence."""
        for k, v in constants.items():
            self.params[k]['value'] *= v

        influence = self._calculate_influence(self.params['infl']['value'])
        return influence * self.params['lr']['value']

    def _init_prev(self, x):
        """Initialize recurrent SOMs."""
        return None

    def _get_bmu(self, activations):
        """Get bmu based on activations."""
        return activations.__getattribute__(self.argfunc)(1)

    def _create_batches(self, X, batch_size, shuffle_data=True):
        """
        Create batches out of a sequence of data.

        This function will append zeros to the end of your data to ensure that
        all batches are even-sized. These are masked out during training.
        """
        if shuffle_data:
            X = shuffle(X)

        if batch_size > X.shape[0]:
            batch_size = X.shape[0]

        max_x = int(np.ceil(X.shape[0] / batch_size))
        X = np.resize(X, (max_x, batch_size, X.shape[-1]))

        return X

    def _propagate(self, x, influences, **kwargs):
        """Propagate a single batch of examples through the network."""
        activation, difference_x = self.forward(x)
        update = self.backward(difference_x, influences, activation)
        # If batch size is 1 we can leave out the call to mean.
        if update.shape[0] == 1:
            self.weights += update[0]
        else:
            self.weights += update.mean(0)

        return activation

    def forward(self, x, **kwargs):
        """
        Get the best matching neurons, and the difference between inputs.

        Note: it might seem like this function can be replaced by a call to
        distance function. This is only true for the regular SOM. other
        SOMs, like the recurrent SOM need more complicated forward pass
        functions.

        parameters
        ==========
        x : numpy array.
            The input vector.

        returns
        =======
        matrices : tuple of matrices.
            A tuple containing the activations and differences between
            neurons and input, respectively.

        """
        return self.distance_function(x, self.weights)

    def backward(self, diff_x, influences, activations, **kwargs):
        """
        Backward pass through the network, including update.

        parameters
        ==========
        diff_x : numpy array
            A matrix containing the differences between the input and neurons.
        influences : numpy array
            A matrix containing the influence each neuron has on each
            other neuron. This is used to calculate the updates.
        activations : numpy array
            The activations each neuron has to each data point. This is used
            to calculate the BMU.

        returns
        =======
        update : numpy array
            A numpy array containing the updates to the neurons.

        """
        bmu = self._get_bmu(activations)
        influence = influences[bmu]
        update = np.multiply(diff_x, influence)
        return update

    def distance_function(self, x, weights):
        """
        Calculate euclidean distance between a batch of input data and weights.

        parameters
        ==========
        X : numpy array.
            The input data.
        weights : numpy array.
            The weights

        returns
        =======
        matrices : tuple of matrices
            The first matrix is a (batch_size * neurons) matrix of
            activation values, containing the response of each neuron
            to each input
            The second matrix is a (batch_size * neurons) matrix containing
            the difference between euch neuron and each input.

        """
        diff = x[:, None, :] - weights[None, :, :]
        activations = np.linalg.norm(diff, axis=2)

        return activations, diff

    def _check_input(self, X):
        """
        Check the input for validity.

        Ensures that the input data, X, is a 2-dimensional matrix, and that
        the second dimension of this matrix has the same dimensionality as
        the weight matrix.
        """
        if X.ndim != 2:
            raise ValueError("Your data is not a 2D matrix. "
                             "Actual size: {0}".format(X.shape))

        if X.shape[1] != self.data_dimensionality:
            raise ValueError("Your data size != weight dim: {0}, "
                             "expected {1}".format(X.shape[1],
                                                   self.data_dimensionality))

    def transform(self, X, batch_size=100, show_progressbar=False):
        """
        Transform input to a distance matrix by measuring the L2 distance.

        parameters
        ==========
        X : numpy array.
            The input data.
        batch_size : int, optional, default 100
            The batch size to use in transformation. This may affect the
            transformation in stateful, i.e. sequential SOMs.
        show_progressbar : bool
            Whether to show a progressbar during transformation.

        returns
        =======
        transformed : numpy array
            A matrix containing the distance from each datapoint to all
            neurons. The distance is normally expressed as euclidean distance,
            but can be any arbitrary metric.

        """
        self._check_input(X)

        batched = self._create_batches(X, batch_size, shuffle_data=False)

        activations = []
        prev = self._init_prev(batched)

        for x in tqdm(batched, disable=not show_progressbar):
            prev = self.forward(x, prev_activation=prev)[0]
            activations.extend(prev)

        activations = np.asarray(activations, dtype=np.float32)
        activations = activations[:X.shape[0]]
        return activations.reshape(X.shape[0], self.num_neurons)

    def predict(self, X, batch_size=1, show_progressbar=False):
        """
        Predict the BMU for each input data.

        parameters
        ==========
        X : numpy array.
            The input data.
        batch_size : int, optional, default 100
            The batch size to use in prediction. This may affect prediction
            in stateful, i.e. sequential SOMs.
        show_progressbar : bool
            Whether to show a progressbar during prediction.

        returns
        =======
        predictions : numpy array
            An array containing the BMU for each input data point.

        """
        dist = self.transform(X, batch_size, show_progressbar)
        res = dist.__getattribute__(self.argfunc)(1)

        return res

    def quantization_error(self, X, batch_size=1):
        """
        Calculate the quantization error.

        Find the the minimum euclidean distance between the units and
        some input.

        parameters
        ==========
        X : numpy array.
            The input data.
        batch_size : int
            The batch size to use for processing.

        returns
        =======
        error : numpy array
            The error for each data point.

        """
        dist = self.transform(X, batch_size)
        res = dist.__getattribute__(self.valfunc)(1)

        return res

    def receptive_field(self,
                        X,
                        identities,
                        max_len=10,
                        threshold=0.9,
                        batch_size=1):
        """
        Calculate the receptive field of the SOM on some data.

        The receptive field is the common ending of all sequences which
        lead to the activation of a given BMU. If a SOM is well-tuned to
        specific sequences, it will have longer receptive fields, and therefore
        gives a better description of the dynamics of a given system.

        parameters
        ==========
        X : numpy array
            Input data.
        identities : list
            A list of symbolic identities associated with each input.
            We enpect this list to be as long as the input data.
        max_len : int, optional, default 10
            The maximum length to attempt to find. Raising this increases
            memory use.
        threshold : float, optional, default .9
            The threshold at which we consider a receptive field
            valid. If at least this proportion of the sequences of a neuron
            have the same suffix, that suffix is counted as acquired by the
            SOM.
        batch_size : int, optional, default 1
            The batch size to use in prediction

        returns
        =======
        receptive_fields : dict
            A dictionary mapping from the neuron id to the found sequences
            for that neuron. The sequences are represented as lists of
            symbols from identities.

        """
        receptive_fields = defaultdict(list)

        predictions = self.predict(X)

        if len(predictions) != len(identities):
            raise ValueError("X and identities are not the same length: "
                             "{0} and {1}".format(len(predictions),
                                                  len(identities)))

        for idx, p in enumerate(predictions.tolist()):
            receptive_fields[p].append(identities[idx+1 - max_len:idx+1])

        rec = {}

        for k, v in receptive_fields.items():
            # if there's only one sequence, we don't know
            # anything abouw how salient it is.
            seq = []
            if len(v) <= 1:
                continue
            else:
                for x in reversed(list(zip(*v))):
                    x = Counter(x)
                    if x.most_common(1)[0][1] / sum(x.values()) > threshold:
                        seq.append(x.most_common(1)[0][0])
                    else:
                        rec[k] = seq
                        break

        return rec

    @classmethod
    def load(cls, path):
        """
        Load a SOM from a JSON file saved with this package.

        parameters
        ==========
        path : str
            The path to the JSON file.

        returns
        =======
        s : cls
            A som of the specified class.

        """
        data = json.load(open(path))

        weights = data['weights']
        weights = np.asarray(weights, dtype=np.float32)

        s = cls(data['num_neurons'],
                data['data_dimensionality'],
                data['params']['lr']['orig'],
                neighborhood=data['params']['infl']['orig'],
                valfunc=data['valfunc'],
                argfunc=data['argfunc'],
                lr_lambda=data['params']['lr']['factor'],
                nb_lambda=data['params']['nb']['factor'])

        s.weights = weights
        s.trained = True

        return s

    def save(self, path):
        """Save a SOM to a JSON file."""
        to_save = {}
        for x in self.param_names:
            attr = self.__getattribute__(x)
            if type(attr) == np.ndarray:
                attr = [[float(x) for x in row] for row in attr]
            elif isinstance(attr, types.FunctionType):
                attr = attr.__name__
            to_save[x] = attr

        json.dump(to_save, open(path, 'w'))
