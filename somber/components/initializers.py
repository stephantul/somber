"""
This file contains initializers.

Initializers can be added to a SOM, and will be
called to initialize the weight to a value at the beginning of training.
"""
import numpy as np


def range_initialization(X, weights):
    """
    Initialize the weights by calculating the range of the data.

    The data range is calculated by reshaping the input matrix to a
    2D matrix, and then taking the min and max values over the columns.

    parameters
    ==========
    X : numpy array
        The input data. The data range is calculated over the last axis.
    weights : numpy array
        The weights. These are not modified in the function.

    returns
    =======
    new_weights : numpy array
        A new version of the weights, initialized to the data range specified
        by X.

    """
    datalen = np.prod(X.shape[:-1])

    # Randomly initialize weights to cover the range of each feature.
    X_ = X.reshape(datalen, X.shape[-1])
    min_val = X_.min(0)
    max_val = X_.max(0)
    data_range = max_val - min_val

    return data_range * np.random.rand(len(weights),
                                       X.shape[-1]) - np.abs(min_val)
