import time
import sys
import cupy as cp
import numpy as np


def np_min(X, axis=None):
    """
    Calculate the min and argmin.

    If no axis is given, this function just returns the min and argmin of X.
    Otherwise, it will return the min and argmin over the specified axis.

    :param X: the input data.
    :param axis: the axis over which to compute the function.
    :return The min and argmin of X
    """
    xp = cp.get_array_module(X)

    if axis is None:
        return xp.min(X), xp.argmin(X)
    else:
        return xp.min(X, axis), xp.argmin(X, axis)


def np_max(X, axis=None):
    """
    Calculate the max and argmax.

    If no axis is given, this function just returns the max and argmax of X.
    Otherwise, it will return the max and argmax over the specified axis.

    :param X: the input data.
    :param axis: the axis over which to compute the function.
    :return The max and argmax of X
    """
    xp = cp.get_array_module(X)

    if axis is None:
        return xp.max(X)
    else:
        return xp.max(X, axis), xp.argmax(X, axis)


def resize(X, new_shape):
    """
    Resizes your numpy arrays.

    Dummy resize function because cupy currently does
    not support direct resizing of arrays.

    :param X: The input data
    :param new_shape: The desired shape of the array.
    Must have the same dimensions as X.
    :return: A resized array.
    """
    xp = cp.get_array_module(X)

    # Difference between actual and desired size
    length_diff = (new_shape[0] * new_shape[1]) - len(X)
    z = xp.zeros((length_diff, X.shape[1]))
    # Pad input data with zeros
    z = xp.concatenate([X, z])

    # Reshape
    return z.reshape(new_shape)


def expo(value, current_step, total_steps):
    """
    Decrease a value X_0 according to an exponential function.

    Lambda is equal to (-2.5 * (current_step / total_steps))

    :param value: The original value.
    :param current_step: The current timestep.
    :param total_steps: The maximum number of steps.
    :return:
    """
    return np.float32(value * np.exp(-2.5 * (current_step / total_steps)))


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
