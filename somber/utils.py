import time
import sys
import numpy as np

from functools import partial


def np_minmax(func1, func2, X, axis=None):

    if axis is None:
        return func1(X)
    else:
        return func1(X, axis), func2(X, axis)

np_min = partial(np_minmax, np.min, np.argmin)
np_max = partial(np_minmax, np.max, np.argmax)


def resize(X, new_shape):
    """
    Dummy resize function because cupy currently does
    not support direct resizing of arrays.

    :param X: The input data
    :param new_shape: The desired shape of the array.
    Must have the same dimensions as X.
    :return: A resized array.
    """

    # Difference between actual and desired size
    length_diff = (new_shape[0] * new_shape[1]) - len(X)

    # Pad input data with zeros
    z = np.concatenate([X, np.zeros((length_diff, X.shape[1]))])

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
    return value * np.exp(-2.5 * (current_step / total_steps))


def static(value, current_step, total_steps):
    """
    Static function: nothing changes.

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


def progressbar(target,
                width=30,
                interval=0.2,
                idx_interval=10,
                use=True,
                mult=1):
    """
    Progressbar, partially borrowed from Keras.

    https://github.com/fchollet/keras/blob/088dbe6866fd51f4e0e64866e442968c17abfa10/keras/utils/generic_utils.py

    :param target: The target of the progressbar, must be some kind of iterable
    :param width: The width of the progressbar in characters
    :param interval: The time interval with which to update the bar.
    :param idx_interval: The index interval with which to update the bar.
    Raise this if the bar is going too fast.
    :param use: Boolean whether to actually use the progressbar
    (for debugging etc.)
    :param mult: The multiplier to multiply the progressbar with,
    useful for accurately displaying batches.
    e.g. instead of displaying 1/1000 batches of size 100,
    you can display 100/100000 items.
    :return: None
    """
    start = time.time()
    last_update = 0
    interval = interval
    total_width = 0

    target = list(target)
    iter_length = len(target)

    prev_total_width = 0

    for current, p in enumerate(target):

        if not use or current % idx_interval:
            yield p
            continue

        now = time.time()

        if (now - last_update) < interval:
            yield p
            continue

        prev_total_width = total_width
        sys.stdout.write("\b" * prev_total_width)
        sys.stdout.write("\r")

        numdigits = int(np.floor(np.log10(iter_length))) + 1
        barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
        bar = barstr % (current * mult, iter_length * mult)
        prog = float(current) / iter_length
        prog_width = int(width * prog)
        if prog_width > 0:
            bar += ('=' * (prog_width-1))
            if current < iter_length:
                bar += '>'
            else:
                bar += '='
        bar += ('.' * (width - prog_width))
        bar += ']'
        sys.stdout.write(bar)
        total_width = len(bar)

        if current:
            time_per_unit = (now - start) / current
        else:
            time_per_unit = 0
        eta = time_per_unit * (iter_length - current)
        info = ''
        if current < iter_length:
            info += ' - ETA: %ds' % eta
        else:
            info += ' - %ds' % (now - start)

        total_width += len(info)
        if prev_total_width > total_width:
            info += ((prev_total_width - total_width) * " ")

        sys.stdout.write(info)
        sys.stdout.flush()

        if current >= iter_length:
            sys.stdout.write("\n")

        last_update = now

        yield p

    else:

        if not use:
            pass
        else:
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(iter_length))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (iter_length * mult, iter_length * mult)
            prog = float(iter_length) / iter_length
            prog_width = int(width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width-1))
                bar += '='
            bar += ('.' * (width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
