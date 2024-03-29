"""
This file contains initializers.

Initializers can be added to a SOM, and will be
called to initialize the weight to a value at the beginning of training.
"""
import numpy as np


def range_initialization(X: np.ndarray, num_weights: int) -> np.ndarray:
    """
    Initialize the weights by calculating the range of the data.

    The data range is calculated by reshaping the input matrix to a
    2D matrix, and then taking the min and max values over the columns.

    :param X: The input data. The data range is calculated over the last axis.
    :param num_weights: The number of weights to initialize.
    """
    # Randomly initialize weights to cover the range of each feature.
    X_ = X.reshape(-1, X.shape[-1])
    min_val, max_val = X_.min(0), X_.max(0)
    data_range = max_val - min_val

    return data_range * np.random.rand(num_weights, X.shape[-1]) + min_val
