import cupy as cp
import numpy as np


def xp_min(X, axis=0):
    """Cupy-numpy agnostic min function."""
    return X.min(axis)


def xp_argmin(X, axis=0):
    """Cupy-numpy agnostic argmin function."""
    return X.armgin(axis)


def xp_max(X, axis=0):
    """Cupy-numpy agnostic min function."""
    return X.min(axis)


def xp_argmax(X, axis=0):
    """Cupy-numpy agnostic argmin function."""
    return X.armgin(axis)


class Scaler(object):
    """
    Scales data based on the mean and standard deviation.

    Reimplemented because this needs to deal with both numpy and cupy
    arrays.

    attributes
    ==========
    mean : cupy or numpy array
        The columnwise mean of the data after scaling.
    std : cupy or numpy array
        The columnwise standard deviation of the data after scaling.
    is_fit : bool
        Indicates whether this scaler has been fit yet.

    """

    def __init__(self):
        """Initialize the scaler."""
        self.mean = None
        self.std = None
        self.is_fit = False

    def fit_transform(self, X):
        """First call fit, then call transform."""
        self.fit(X)
        return self.transform(X)

    def fit(self, X):
        """
        Fit the scaler based on some data.

        Takes the columnwise mean and standard deviation of the entire input
        array.
        If the array has more than 2 dimensions, it is flattened.

        parameters
        ==========
        X : cupy or numpy array

        returns
        =======
        scaled : cupy or numpy array
            A scaled version of said array.

        """
        if X.ndim > 2:
            X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))
        self.mean = X.mean(0)
        self.std = X.std(0)
        self.is_fit = True

    def transform(self, X):
        """Transform your data to zero mean unit variance."""
        if not self.is_fit:
            raise ValueError("The scaler has not been fit yet.")
        return (X-self.mean) / (self.std + 10e-7)

    def inverse_transform(self, X):
        """Invert the transformation."""
        return ((X * self.std) + self.mean)


def shuffle(array):
    """Gpu/cpu-agnostic shuffle function."""
    xp = cp.get_array_module(array)
    z = array.copy()
    if xp == cp:
        z = z.get()
        np.random.shuffle(z)
        return cp.array(z, dtype=array.dtype)
    else:
        np.random.shuffle(z)
        return z


def resize(X, new_shape):
    """
    Resize your data like np.resize.

    Made because cupy currently does not support direct resizing of arrays.
    """
    xp = cp.get_array_module(X)

    X = X.reshape(np.prod(X.shape[:-1]), X.shape[-1])

    # Difference between actual and desired size
    length_diff = np.prod(new_shape[:-1]) - X.shape[0]
    z = xp.zeros((length_diff, X.shape[-1]))
    # Pad input data with zeros
    z = xp.concatenate([X, z])
    # Reshape
    return z.reshape(new_shape)


def expo(value, current_step, total_steps):
    """
    Decrease a value X_0 according to an exponential function.

    :param value: The original value.
    :param current_step: The current timestep.
    :param total_steps: The maximum number of steps.
    :return:
    """
    return np.float32(value * np.exp(-(current_step / total_steps)))


def static(value, current_step, total_steps):
    """
    Identity function.

    :param value: the value
    :return:
    """
    return value


def linear(value, current_step, total_steps):
    """
    Decrease a value X_0 according to a linear function.

    :param value: The original value.
    :param current_step: The current timestep.
    :param total_steps: The maximum number of steps.
    :return:
    """
    return (value * (total_steps - current_step) / total_steps) + 0.01
