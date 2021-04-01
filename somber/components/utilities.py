"""Utility functions."""
import numpy as np


class Scaler(object):
    """Scales data based on the mean and standard deviation."""

    def __init__(self):
        """Initialize the scaler."""
        self.mean = None
        self.std = None
        self.is_fit = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """First call fit, then call transform."""
        self.fit(X)
        return self.transform(X)

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the scaler based on some data.

        Takes the columnwise mean and standard deviation of the entire input
        array.
        If the array has more than 2 dimensions, it is flattened.
        """
        if X.ndim > 2:
            X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))
        self.mean = X.mean(0)
        self.std = X.std(0)
        self.is_fit = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform your data to zero mean unit variance."""
        if not self.is_fit:
            raise ValueError("The scaler has not been fit yet.")
        return (X - self.mean) / (self.std + 10e-7)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the transformation."""
        return (X * self.std) + self.mean


def shuffle(array: np.ndarray) -> np.ndarray:
    """Gpu/cpu-agnostic shuffle function."""
    return np.random.permutation(array)
