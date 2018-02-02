"""Utility functions."""
import numpy as np


class Scaler(object):
    """
    Scales data based on the mean and standard deviation.

    attributes
    ==========
    mean : numpy array
        The columnwise mean of the data after scaling.
    std : numpy array
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
        X : numpy array

        returns
        =======
        scaled : numpy array
            A scaled version of said array.

        """
        if X.ndim > 2:
            X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))
        self.mean = X.mean(0)
        self.std = X.std(0)
        self.is_fit = True
        return self

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
    z = array.copy()
    np.random.shuffle(z)
    return z
